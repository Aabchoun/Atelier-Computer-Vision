"""Coordinator: triggers the agents, resamples their outputs onto the common grid, fuses them,
handles failures (fallback on the last known layers) and publishes risk map, uncertainty and alerts.

Flow of one run (plain Python calls for now; Ray / LangGraph / Redis only once this works):
    1. perception agents (fire -> fire fusion -> smoke, vegetation, burned area, environment)
    2. forecasting agents on the datacube (spread prediction, simulation)
    3. ensemble -> risk, disagreement, uncertain zones, alerts, wind check
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from firespread.agents import FORECAST_AGENTS, PERCEPTION_AGENTS, build_agents
from firespread.agents.vegetation import BUILT_UP
from firespread.common.agent import Agent, AgentInputError
from firespread.common.indices import angle_difference, wind_direction_deg
from firespread.common.message import AgentMessage
from firespread.common.raster_ops import dilate
from firespread.common.scene import RawScene, SceneDatacube
from firespread.coordinator.fusion import staleness_weight, weighted_nanmean
from firespread.coordinator.report import Report

FIRE_SOURCES = ("fire_s2", "fire_firms")  # observed at about the same time: weighted together
NEWER_FIRE_SOURCES = ("fire_firms_new",)  # detections after the optical snapshot: added on top
FORECAST_LAYERS = ("spread_prob", "sim_prob")


@dataclass
class PipelineResult:
    cube: SceneDatacube
    report: Report

    def save(self, directory: str | Path, quicklook: bool = True) -> Path:
        directory = Path(directory)
        self.cube.save(directory / "datacube")
        self.report.save(directory)
        if quicklook:
            try:
                from firespread.viz import save_quicklook

                save_quicklook(self.cube, directory / "quicklook.png")
            except ImportError:  # matplotlib missing: the run is still saved
                pass
        return directory


class Coordinator:
    def __init__(self, config: dict[str, Any], agents: dict[str, Agent] | None = None) -> None:
        self.config = config
        self.cfg = config["coordinator"]
        self.agents = build_agents(config) if agents is None else agents
        # State kept between runs over the SAME area (use one Coordinator per area / fire):
        # layers of previous runs, used when an agent fails or its scene is too cloudy ...
        self.last_known: dict[str, AgentMessage] = {}
        # ... and the burning cells of previous runs, to keep the burned area up to date between optical passes.
        self.fire_history: list[tuple[datetime, np.ndarray]] = []

    # ------------------------------------------------------------------ run
    def run(self, raw: RawScene) -> PipelineResult:
        forecast_time = raw.latest_time()
        cube = SceneDatacube(raw.grid)
        report = Report(scene=raw.name, forecast_time=forecast_time)
        n_points = len(raw.firms) if raw.firms is not None else 0
        report.add("info", "coordinator", f"Run for scene '{raw.name}' at {forecast_time:%Y-%m-%d %H:%M} UTC, {n_points} FIRMS hotspots.")

        for name in PERCEPTION_AGENTS:
            self._run_agent(name, raw, cube, report)
            if name == "fire":
                self._fuse_fire(cube, report, forecast_time)

        if not cube.has("fire"):
            report.add("error", "coordinator", "No fire layer (no optical scene, no FIRMS, no previous layer): forecast skipped.")
            return PipelineResult(cube, report)
        if self.cfg.get("accumulate_burned", True):
            self._accumulate_burned(cube, report, forecast_time)

        for name in FORECAST_AGENTS:
            self._run_agent(name, raw, cube, report)
        self._publish(cube, report)
        self.fire_history.append((cube.message("fire").acquisition_time, cube.get("fire") >= self.cfg["fire_threshold"]))
        return PipelineResult(cube, report)

    # --------------------------------------------------------------- agents
    def _run_agent(self, name: str, raw: RawScene, cube: SceneDatacube, report: Report) -> None:
        agent = self.agents.get(name)
        if agent is None:
            report.add("info", name, "Agent disabled.")
            return
        try:
            messages = agent.run(raw, cube)
        except AgentInputError as exc:  # expected, e.g. no Sentinel-2 pass today (revisit gap)
            report.add("info", name, f"Agent skipped: {exc}.")
            self._fallback_all(agent, cube, report)
            return
        except Exception as exc:  # one failing agent must not stop the system (robustness requirement)
            if self.cfg.get("strict", False):
                raise
            report.add("warning", name, f"Agent failed: {type(exc).__name__}: {exc}")
            self._fallback_all(agent, cube, report)
            return
        for message in messages:
            self._ingest(message, cube, report)

    def _fallback_all(self, agent: Agent, cube: SceneDatacube, report: Report) -> None:
        for layer in agent.produces:
            if self._fallback(layer, cube):
                report.add("info", "coordinator", f"'{layer}': using last known layer from {self.last_known[layer].acquisition_time:%Y-%m-%d %H:%M}.")

    def _ingest(self, msg: AgentMessage, cube: SceneDatacube, report: Report) -> None:
        resampling = self.cfg.get("resampling", {})
        method = resampling.get(msg.layer, resampling.get("default", "mean"))
        on_grid = replace(msg, data=cube.grid.resample(msg.data, method), resolution=cube.grid.resolution)

        unreliable = msg.confidence < self.cfg["min_confidence"] or msg.cloud_fraction > self.cfg["max_cloud_fraction"]
        if unreliable:
            quality = f"confidence {msg.confidence:.2f}, cloud {msg.cloud_fraction:.0%}"
            if self._fallback(msg.layer, cube):
                previous = self.last_known[msg.layer]
                report.add("warning", msg.agent, f"'{msg.layer}' unreliable ({quality}): using last known layer from {previous.acquisition_time:%Y-%m-%d %H:%M}.")
                return
            report.add("warning", msg.agent, f"'{msg.layer}' unreliable ({quality}) and no previous layer: used anyway.")
        else:
            self.last_known[msg.layer] = on_grid

        missing = msg.extras.get("missing_inputs")
        if missing:
            report.add("warning", msg.agent, f"'{msg.layer}' computed without {', '.join(missing)} (neutral values used).")
        cube.put(on_grid)

    def _fallback(self, layer: str, cube: SceneDatacube) -> bool:
        previous = self.last_known.get(layer)
        if previous is None:
            return False
        cube.put(replace(previous, stale=True))
        return True

    def _accumulate_burned(self, cube: SceneDatacube, report: Report, forecast_time: datetime) -> None:
        """Cells that burned in previous runs AFTER the last burned-area map are burned too.

        Between two optical passes the dNBR map gets older while the fire moves; without this the
        simulation would let the fire spread back into fuel that is already gone.
        """
        burned = cube.message("burned")
        since = burned.acquisition_time if burned is not None else datetime.min
        past = [mask for t, mask in self.fire_history if since < t < forecast_time and mask.shape == cube.grid.shape]
        if not past:
            return
        previous_fire = np.logical_or.reduce(past)
        already_burned = np.nan_to_num(burned.data) >= 0.5 if burned is not None else np.zeros(cube.grid.shape, bool)
        if burned is not None:
            data = np.maximum(np.nan_to_num(burned.data), previous_fire.astype(np.float32))
            cube.put(replace(burned, data=data, extras={**burned.extras, "fire_history_runs": len(past)}))
        else:
            cube.put(AgentMessage(
                agent="coordinator", layer="burned", acquisition_time=forecast_time, sensor="fire history",
                crs=cube.grid.crs, bbox=cube.grid.bbox, resolution=cube.grid.resolution, confidence=0.5,
                extras={"fire_history_runs": len(past)}, data=previous_fire.astype(np.float32),
            ))
        added = int((previous_fire & ~already_burned).sum())
        report.add("info", "coordinator", f"Burned area completed with the fire of {len(past)} previous run(s): +{added} cells.")

    # --------------------------------------------------------------- fusion
    def _fuse_fire(self, cube: SceneDatacube, report: Report, forecast_time: datetime) -> None:
        """fire = per-pixel weighted mean of the optical fire mask and the FIRMS hotspots acquired
        around the same time (NaN-aware), then max-merged with the FIRMS hotspots acquired later.

        Under clouds only FIRMS counts. A concurrent FIRMS hotspot in a clear optical pixel without
        fire is down-weighted (likely false alarm or already extinguished). A later hotspot is newer
        information than the optical snapshot, so the snapshot cannot reject it.
        """
        sources = [cube.message(layer) for layer in FIRE_SOURCES if cube.has(layer)]
        newer = [cube.message(layer) for layer in NEWER_FIRE_SOURCES if cube.has(layer)]
        if not sources and not newer:
            return
        tau = self.cfg["staleness_tau_days"]
        weights = [staleness_weight(m.confidence, m.acquisition_time, forecast_time, tau) for m in sources]
        fused = np.zeros(cube.grid.shape, np.float32)
        if sources:
            fused = np.nan_to_num(weighted_nanmean([m.data for m in sources], weights), nan=0.0)
        for message in newer:
            fused = np.maximum(fused, np.nan_to_num(message.data))
        sources += newer
        weights += [staleness_weight(m.confidence, m.acquisition_time, forecast_time, tau) for m in newer]
        cube.put(
            AgentMessage(
                agent="coordinator",
                layer="fire",
                acquisition_time=max(m.acquisition_time for m in sources),
                sensor=" + ".join(dict.fromkeys(m.sensor for m in sources)),
                crs=cube.grid.crs,
                bbox=cube.grid.bbox,
                resolution=cube.grid.resolution,
                confidence=max(weights),
                cloud_fraction=min(m.cloud_fraction for m in sources),
                extras={"weights": {m.layer: round(w, 3) for m, w in zip(sources, weights)}},
                data=fused,
            )
        )
        burning = int((fused >= self.cfg["fire_threshold"]).sum())
        detail = ", ".join(f"{m.layer} w={w:.2f}{' (stale)' if m.stale else ''}" for m, w in zip(sources, weights))
        report.add("info", "coordinator", f"Fire layer fused from {detail}: {burning} burning cells.")

    def _publish(self, cube: SceneDatacube, report: Report) -> None:
        fusion = self.cfg["fusion"]
        forecasts = [cube.message(layer) for layer in FORECAST_LAYERS if cube.has(layer)]
        if not forecasts:
            report.add("error", "coordinator", "No forecast available: no risk map.")
            return
        risk = np.nan_to_num(weighted_nanmean([m.data for m in forecasts], [m.confidence for m in forecasts]))
        confidence = float(np.mean([m.confidence for m in forecasts]))
        cube.put(self._derived("risk", risk, forecasts[0], confidence))
        if len(forecasts) == 2:
            disagreement = np.abs(forecasts[0].data - forecasts[1].data)
            cube.put(self._derived("disagreement", disagreement, forecasts[0], confidence))
            cube.put(self._derived("uncertain", disagreement >= fusion["disagreement_threshold"], forecasts[0], confidence))
        self._alerts(cube, report, risk, horizon=forecasts[0].extras.get("horizon_days", 1))
        self._wind_check(cube, report)

    @staticmethod
    def _derived(layer: str, data: np.ndarray, ref: AgentMessage, confidence: float) -> AgentMessage:
        return replace(ref, agent="coordinator", layer=layer, sensor="ensemble", data=np.asarray(data, np.float32),
                       confidence=confidence, extras={}, stale=False, data_ref=None)

    def _alerts(self, cube: SceneDatacube, report: Report, risk: np.ndarray, horizon: int) -> None:
        fusion = self.cfg["fusion"]
        cell_ha = cube.grid.cell_area_ha
        burning = cube.get("fire") >= self.cfg["fire_threshold"]
        at_risk = (risk >= fusion["risk_threshold"]) & ~burning
        report.summary["burning_area_ha"] = round(float(burning.sum() * cell_ha), 1)
        report.summary["new_area_at_risk_ha"] = round(float(at_risk.sum() * cell_ha), 1)
        if at_risk.any():
            report.add("alert", "coordinator", f"{at_risk.sum() * cell_ha:.0f} ha outside the current fire at risk >= {fusion['risk_threshold']} within {horizon} day(s).")
        if cube.has("uncertain"):
            uncertain = cube.get("uncertain") > 0.5
            report.summary["uncertain_area_ha"] = round(float(uncertain.sum() * cell_ha), 1)
            if uncertain.any():
                report.add("warning", "coordinator", f"{uncertain.sum() * cell_ha:.0f} ha flagged uncertain: prediction and simulation disagree.")
        landcover = cube.get("landcover")
        if landcover is not None:
            near_fire = dilate(risk >= fusion["built_up_risk_threshold"], iterations=1)
            threatened = (np.rint(landcover) == BUILT_UP) & near_fire
            report.summary["built_up_threatened_ha"] = round(float(threatened.sum() * cell_ha), 1)
            if threatened.any():
                report.add("alert", "coordinator", f"Built-up area threatened: {threatened.sum() * cell_ha:.0f} ha next to cells at risk >= {fusion['built_up_risk_threshold']}.")

    def _wind_check(self, cube: SceneDatacube, report: Report) -> None:
        """The plume direction from the Smoke agent should roughly follow the ERA5 wind."""
        smoke = cube.message("smoke")
        if smoke is None or smoke.stale or "plume_direction_deg" not in smoke.extras or not cube.has("wind_u"):
            return
        plume = smoke.extras["plume_direction_deg"]
        wind = float(wind_direction_deg(np.nanmean(cube.get("wind_u")), np.nanmean(cube.get("wind_v"))))
        diff = angle_difference(plume, wind)
        report.summary["plume_vs_wind_deg"] = round(diff, 1)
        ok = diff <= self.cfg["fusion"]["wind_check_max_deg"]
        report.add("info" if ok else "warning", "smoke",
                   f"Plume points to {plume:.0f} deg, ERA5 wind blows to {wind:.0f} deg (difference {diff:.0f} deg)."
                   + ("" if ok else " Check the wind layer or the smoke mask."))
