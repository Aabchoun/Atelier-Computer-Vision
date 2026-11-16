"""Calibration of the cellular-automaton coefficients against the observed next-day active fire.

One CalibrationCase = the datacube of one forecast day (initial state, fuel, terrain, wind) and the
FIRMS hotspots of the next day. The objective is the micro-averaged F1 of `sim_prob >= 0.5` over all
cases, i.e. exactly what the benchmark reports. Calibrate on early days, test on later ones.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import numpy as np

from firespread.agents.simulation.cellular_automaton import CAParams, initial_state, run_monte_carlo, spread_probabilities
from firespread.common.scene import SceneDatacube

ENV_DEFAULTS = {"p_veg": 0.0, "p_den": 0.0, "elevation": 0.0, "wind_u": 0.0, "wind_v": 0.0}


@dataclass
class CalibrationCase:
    state0: np.ndarray
    env: dict[str, np.ndarray]  # p_veg, p_den, elevation, wind_u, wind_v
    truth: np.ndarray  # bool, active fire on the next day
    cell_size: float


def case_from_cube(cube: SceneDatacube, truth: np.ndarray, fire_threshold: float = 0.5) -> CalibrationCase:
    def layer(name: str, default: float) -> np.ndarray:
        data = cube.get(name)
        return np.full(cube.grid.shape, default, np.float32) if data is None else data

    state0 = initial_state(cube.get("fire"), layer("burned", 0.0), layer("burnable", 1.0), fire_threshold)
    env = {name: layer(name, default) for name, default in ENV_DEFAULTS.items()}
    return CalibrationCase(state0, env, np.asarray(truth) > 0.5, cube.grid.resolution)


def score(cases: list[CalibrationCase], params: CAParams, steps_per_day: int, n_runs: int = 10, seed: int = 0, threshold: float = 0.5) -> dict[str, float]:
    """Micro-averaged precision / recall / F1 of the simulated next-day active fire."""
    tp = fp = fn = 0
    for case in cases:
        probs = spread_probabilities(**case.env, cell_size=case.cell_size, params=params)
        result = run_monte_carlo(case.state0, probs, horizon_days=1, steps_per_day=steps_per_day, n_runs=n_runs, seed=seed)
        active = result.active_prob >= threshold
        tp += int((active & case.truth).sum())
        fp += int((active & ~case.truth).sum())
        fn += int((~active & case.truth).sum())
    nan = float("nan")
    return {
        "precision": tp / (tp + fp) if tp + fp else nan,
        "recall": tp / (tp + fn) if tp + fn else nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else nan,
    }


def calibrate(cases: list[CalibrationCase], search: dict[str, list[float]], steps_options: list[int], n_runs: int = 10, seed: int = 0) -> tuple[CAParams, int, dict[str, float], list[dict]]:
    """Grid search over CAParams fields in `search` and CA steps per day. Returns the best
    (params, steps_per_day, scores) and every tried combination."""
    tried: list[dict] = []
    best: tuple[CAParams, int, dict[str, float]] | None = None
    for steps in steps_options:
        for values in itertools.product(*search.values()):
            params = replace(CAParams(), **dict(zip(search, values)))
            scores = score(cases, params, steps, n_runs, seed)
            tried.append({"steps_per_day": steps, **dict(zip(search, values)), **scores})
            if best is None or np.nan_to_num(scores["f1"], nan=-1.0) > np.nan_to_num(best[2]["f1"], nan=-1.0):
                best = (params, steps, scores)
    assert best is not None, "empty search space"
    return best[0], best[1], best[2], tried
