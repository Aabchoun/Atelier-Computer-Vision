"""Simulation agent: physics-inspired cellular automaton run on the shared datacube."""

from __future__ import annotations

import numpy as np

from firespread.agents.simulation.cellular_automaton import CAParams, initial_state, run_monte_carlo, spread_probabilities
from firespread.common.agent import Agent, AgentInputError
from firespread.common.message import AgentMessage
from firespread.common.scene import RawScene, SceneDatacube


class SimulationAgent(Agent):
    """Outputs:
    sim_prob       - P(cell has active fire on the last forecast day), comparable to spread_prob
    sim_reach_prob - P(cell burning or burned by the end of the horizon)

    Missing drivers (ablation, failed agent) fall back to neutral values and are reported:
    burnable=1, burned=0, p_veg=0, p_den=0, flat terrain, no wind.
    """

    name = "simulation"
    produces = ("sim_prob", "sim_reach_prob")

    def run(self, raw: RawScene, cube: SceneDatacube) -> list[AgentMessage]:
        fire_msg = cube.message("fire")
        if fire_msg is None:
            raise AgentInputError("no fused fire layer in the datacube")
        missing: list[str] = []

        def layer(name: str, default: float) -> np.ndarray:
            data = cube.get(name)
            if data is None:
                missing.append(name)
                return np.full(cube.grid.shape, default, dtype=np.float32)
            return data

        state0 = initial_state(fire_msg.data, layer("burned", 0.0), layer("burnable", 1.0), self.config.get("fire_threshold", 0.5))
        params = CAParams(**self.config.get("params", {}))
        probs = spread_probabilities(
            layer("p_veg", 0.0), layer("p_den", 0.0), layer("elevation", 0.0), layer("wind_u", 0.0), layer("wind_v", 0.0),
            cell_size=cube.grid.resolution, params=params,
        )
        horizon = int(self.config.get("horizon_days", 1))
        result = run_monte_carlo(
            state0, probs,
            horizon_days=horizon,
            steps_per_day=int(self.config.get("steps_per_day", 4)),
            n_runs=int(self.config.get("n_runs", 30)),
            seed=int(self.config.get("seed", 0)),
        )
        common = dict(
            acquisition_time=fire_msg.acquisition_time,
            sensor="cellular automaton",
            confidence=self.config.get("confidence", 0.5),
            extras={"model": "alexandridis-ca", "horizon_days": horizon, "missing_inputs": missing},
        )
        return [
            self.message(cube.grid, "sim_prob", result.active_prob, **common),
            self.message(cube.grid, "sim_reach_prob", result.reach_prob, **common),
        ]
