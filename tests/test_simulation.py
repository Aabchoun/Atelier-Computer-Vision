import numpy as np

from firespread.agents.simulation.cellular_automaton import (
    BURNED,
    BURNING,
    NON_BURNABLE,
    UNBURNED,
    initial_state,
    run_monte_carlo,
    spread_probabilities,
)

N = 41
CENTRE = N // 2


def ignition():
    state = np.full((N, N), UNBURNED, np.int8)
    state[CENTRE, CENTRE] = BURNING
    return state


def probs(wind_u=0.0, wind_v=0.0, elevation=None):
    zeros = np.zeros((N, N), np.float32)
    return spread_probabilities(zeros, zeros, zeros if elevation is None else elevation, wind_u, wind_v, cell_size=375.0)


def reached(state, p, steps=10):
    return run_monte_carlo(state, p, horizon_days=1, steps_per_day=steps, n_runs=20, seed=1).reach_prob > 0.5


def test_wind_pushes_fire_downwind():
    rows, cols = np.nonzero(reached(ignition(), probs(wind_u=8.0)))  # wind blowing towards the east
    assert cols.mean() > CENTRE + 2
    assert abs(rows.mean() - CENTRE) < 1.5


def test_fire_spreads_faster_upslope():
    elevation = (np.arange(N, dtype=np.float32)[:, None] * 50.0).repeat(N, axis=1)  # higher to the south
    rows, _ = np.nonzero(reached(ignition(), probs(elevation=elevation)))
    assert rows.mean() > CENTRE + 1


def test_non_burnable_cells_never_burn():
    state = ignition()
    state[:, CENTRE + 3] = NON_BURNABLE  # firebreak east of the ignition
    result = run_monte_carlo(state, probs(wind_u=8.0), horizon_days=1, steps_per_day=10, n_runs=10, seed=0)
    assert result.reach_prob[:, CENTRE + 3].max() == 0


def test_initial_state_from_layers():
    fire = np.array([[0.9, 0.0, 0.0]])
    burned = np.array([[1.0, 1.0, 0.0]])
    burnable = np.array([[1.0, 1.0, 0.0]])
    assert initial_state(fire, burned, burnable).tolist() == [[BURNING, BURNED, NON_BURNABLE]]
