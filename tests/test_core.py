import hashlib, json, os

import numpy as np
import pytest

from lander.env import World, GOAL_WORLD, Flight, run_episode, make_env, boulder_spec, inputs, lander_xy
from lander.policy import load_policy, NumpyMLP
from lander.teacher import Teacher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The task is frozen: copied unchanged from ../lunarlander-curriculum on 2026-10-30. teacher.py is part of the task
# because env.py's route sensors (inputs 14-15) call teacher.target_x with route_params.json.
FROZEN = {
    'lander/env.py': 'dd1c20453c0c389db3844f3b167bd726c3529d2b4241b0d742bb75098bfd2754',
    'lander/route_params.json': 'beb1a0c56d78368e0ac3835dfa144018778e40ea8736f4e3a4c26daa069e31ca',
    'lander/teacher.py': '4a5d61e3ba779a97027a19f71bbe7350461ad1cc72ebe366fe61194b527abfc6',
}


@pytest.mark.parametrize('path', list(FROZEN))
def test_task_is_frozen(path):
    assert hashlib.sha256(open(os.path.join(ROOT, path), 'rb').read()).hexdigest() == FROZEN[path]


@pytest.mark.parametrize('world', [World(), GOAL_WORLD, World(steady=1.0, support=0.5)])
def test_same_seed_same_flight(world):
    env = make_env()
    a = run_episode(Teacher(), 123, world, env=env)
    run_episode(Teacher(), 999, GOAL_WORLD, env=env)              # reuse the env in between
    b = run_episode(Teacher(), 123, world, env=env)
    c = run_episode(Teacher(), 123, world)                         # and a brand-new env
    assert a['reward'] == b['reward'] == c['reward'] and a['steps'] == b['steps'] == c['steps']


def test_wind_depends_on_seed():
    assert run_episode(Teacher(), 1, World())['reward'] != run_episode(Teacher(), 2, World())['reward']


def test_boulder_hit_is_a_crash():
    seed = next(s for s in range(10000) if abs(boulder_spec(s)[0] - 10) < 0.3)
    r = run_episode(lambda v, f: 0, seed, World(wind=0, turbulence=0, boulder=True))
    assert r['outcome'] == 'boulder' and not r['landed']


def test_seed_ranges_do_not_overlap():
    from lander.seeds import SEED_RANGES, GOAL_BLOCK_1, GOAL_BLOCK_2, BASELINE_BLOCK, TEST, ANALYSIS
    spans = sorted(SEED_RANGES.values())
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    assert GOAL_BLOCK_1 + 1000 <= GOAL_BLOCK_2 and GOAL_BLOCK_2 + 1000 <= BASELINE_BLOCK < ANALYSIS
    assert all(TEST <= s < ANALYSIS for s in (GOAL_BLOCK_1, GOAL_BLOCK_2))


def test_training_seeds_stay_in_the_training_range():
    from lander.seeds import CLONE
    from rl.ppo import _new_flight
    import argparse
    env = make_env()
    counters = [999_999] * 8
    f = _new_flight(env, 7, 7, 8, counters, GOAL_WORLD, None, np.random.default_rng(0))
    assert f.seed < CLONE                                          # last slot, last seed: still training range


class Spy:
    """A policy that records what it was given, and checks it never needs the flight object."""
    def __init__(self, inner):
        self.inner, self.seen = inner, []

    def __call__(self, v, flight=None):
        self.seen.append((np.array(v), flight))
        return self.inner(v, None)                                 # the real policy gets no flight at all


@pytest.mark.parametrize('spec', ['teacher:v2', 'models/bc_v2_128.json'])
def test_policy_sees_only_inputs(spec):
    """The policy's inputs equal inputs() computed from the observation history alone, and the policy flies
    the same flight without access to the env (flight=None)."""
    pol = load_policy(os.path.join(ROOT, spec) if spec.endswith('.json') else spec)
    env = make_env()
    spy = Spy(pol)
    ref = run_episode(lambda v, f: pol(v, f), 4242, GOAL_WORLD, env=env)
    got = run_episode(spy, 4242, GOAL_WORLD, env=env)
    assert ref['reward'] == got['reward'] and ref['steps'] == got['steps']
    # recompute every input vector from observations + own actions only
    f = Flight(env, 4242, GOAL_WORLD)
    for v, _ in spy.seen[:-1]:
        assert np.array_equal(v, f.inputs())
        assert len(v) == 16
        if f.step(pol(v, None)):
            break


def test_numpy_policy_matches_torch_actor():
    import torch
    from rl.ppo import AC
    net = AC((32, 32))
    with torch.no_grad():
        net.mean.uniform_(-1, 1); net.std.uniform_(0.5, 2)
    d = net.export()
    pol = NumpyMLP(d)
    X = np.random.default_rng(0).normal(0, 2, (3000, 16))
    with torch.no_grad():
        t = net.pi(net.norm(torch.tensor(X, dtype=torch.float32))).argmax(1).numpy()
    assert (np.array([pol(x) for x in X]) == t).mean() > 0.999    # float32 vs float64 ties only


def test_old_clone_loads_identically():
    from lander.clone import MLP
    d = json.load(open(os.path.join(ROOT, 'models/seed_mlp16.json')))
    a, b = MLP.from_dict(d), NumpyMLP(d)
    X = np.random.default_rng(1).normal(0, 1, (2000, 16))
    assert all(a(x) == b(x) for x in X)
