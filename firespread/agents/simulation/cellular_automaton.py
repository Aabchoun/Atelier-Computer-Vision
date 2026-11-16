"""Cellular automaton of Alexandridis et al. (2008). Each grid cell is a *cell agent*.

States: NON_BURNABLE, UNBURNED, BURNING, BURNED. At each step a BURNING cell becomes BURNED and
tries to ignite each UNBURNED neighbour with probability

    p_burn = p_h * (1 + p_veg) * (1 + p_den) * p_w * p_s
    p_w    = exp(c1 * V) * exp(c2 * V * (cos(theta) - 1))   V: wind speed (m/s), theta: angle between
                                                              wind and propagation direction
    p_s    = exp(a * theta_s)                                theta_s: slope angle (deg) from the
                                                              burning cell up to the neighbour

The default coefficients are the paper's (Spetses 1990, small cells). At 375 m cells they MUST be
re-calibrated on observed fires: see calibration.py and scripts/calibrate_ca.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from firespread.common.raster_ops import NEIGHBOURS_8, shift

NON_BURNABLE, UNBURNED, BURNING, BURNED = 0, 1, 2, 3


@dataclass(frozen=True)
class CAParams:
    p_h: float = 0.58
    a: float = 0.078
    c1: float = 0.045
    c2: float = 0.131


@dataclass
class SimulationResult:
    active_prob: np.ndarray  # P(cell burning at some step of the last forecast day)
    reach_prob: np.ndarray  # P(cell burning or burned at the end of the horizon)


def initial_state(fire: np.ndarray, burned: np.ndarray, burnable: np.ndarray, fire_threshold: float = 0.5) -> np.ndarray:
    """CA state from the datacube: burning where fire >= threshold, burned scar, non-burnable cells."""
    state = np.where(np.nan_to_num(burnable, nan=1.0) >= 0.5, UNBURNED, NON_BURNABLE).astype(np.int8)
    state[np.nan_to_num(burned) >= 0.5] = BURNED
    state[np.nan_to_num(fire) >= fire_threshold] = BURNING
    return state


def spread_probabilities(p_veg, p_den, elevation, wind_u, wind_v, cell_size: float, params: CAParams = CAParams()) -> np.ndarray:
    """(8, H, W) array: probability that a burning cell at (r - dr, c - dc) ignites cell (r, c)."""
    shape = np.shape(p_veg)

    def field(x):
        return np.nan_to_num(np.broadcast_to(np.asarray(x, dtype=np.float32), shape)).astype(np.float32)

    p_veg, p_den, elevation, wind_u, wind_v = map(field, (p_veg, p_den, elevation, wind_u, wind_v))
    speed = np.hypot(wind_u, wind_v)
    base = params.p_h * (1 + p_veg) * (1 + p_den)
    probs = np.empty((len(NEIGHBOURS_8), *shape), dtype=np.float32)
    for k, (d_row, d_col) in enumerate(NEIGHBOURS_8):
        length = np.hypot(d_row, d_col)
        east, north = d_col / length, -d_row / length  # propagation direction (rows increase southwards)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_theta = np.where(speed > 0, (wind_u * east + wind_v * north) / speed, 1.0)
        p_w = np.exp(params.c1 * speed) * np.exp(params.c2 * speed * (cos_theta - 1))
        source_elevation = shift(elevation, d_row, d_col, fill=np.nan)
        slope_deg = np.degrees(np.arctan((elevation - source_elevation) / (cell_size * length)))
        p_s = np.exp(params.a * np.nan_to_num(slope_deg))
        probs[k] = np.clip(base * p_w * p_s, 0.0, 1.0)
    return probs


def step(state: np.ndarray, probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    burning = state == BURNING
    p_not_ignited = np.ones(state.shape, dtype=np.float32)
    for k, (d_row, d_col) in enumerate(NEIGHBOURS_8):
        from_burning = shift(burning, d_row, d_col, fill=False)
        p_not_ignited *= np.where(from_burning, 1.0 - probs[k], 1.0)
    ignite = (state == UNBURNED) & (rng.random(state.shape) < 1.0 - p_not_ignited)
    new = state.copy()
    new[burning] = BURNED
    new[ignite] = BURNING
    return new


def simulate(state0: np.ndarray, probs: np.ndarray, n_steps: int, seed: int = 0) -> np.ndarray:
    """One stochastic run; returns the (n_steps + 1, H, W) history (for animations in the demo)."""
    rng = np.random.default_rng(seed)
    history = [state0]
    for _ in range(n_steps):
        history.append(step(history[-1], probs, rng))
    return np.stack(history)


def run_monte_carlo(state0: np.ndarray, probs: np.ndarray, horizon_days: int, steps_per_day: int, n_runs: int, seed: int = 0) -> SimulationResult:
    rng = np.random.default_rng(seed)
    total = horizon_days * steps_per_day
    last_day_start = total - steps_per_day
    active_count = np.zeros(state0.shape, dtype=np.float32)
    reach_count = np.zeros(state0.shape, dtype=np.float32)
    for _ in range(n_runs):
        state = state0.copy()
        active = np.zeros(state0.shape, dtype=bool)
        for s in range(total):
            if s >= last_day_start:
                active |= state == BURNING
            state = step(state, probs, rng)
        active_count += active
        reach_count += (state == BURNING) | (state == BURNED)
    return SimulationResult(active_count / n_runs, reach_count / n_runs)
