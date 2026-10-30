# Lander 98: a lunar lander that lands 98% of flights in wind, turbulence and past a boulder

**Live site (log, curriculum, model, speed, podcast): https://jca374.github.io/lunarlander-98**

Gymnasium LunarLander-v2 (discrete), wind 15, turbulence 1.5, a boulder on the approach. Goal: ≥ 98% of 1,000 unseen flights
land between the flags (95% Wilson interval reported), confirmed on a second fresh 1,000; deadline Wednesday 2026-11-04 17:00.
The previous project ([lunarlander-curriculum](https://github.com/JCA374/lunarlander-curriculum), NEAT) stalled at ~85%.

- `lander/env.py`, `lander/route_params.json`, `lander/teacher.py`: the task, frozen (sha256-tested in `tests/test_core.py`).
- `rl/ppo.py`: PPO with a curriculum, checkpoints, checks on fresh seeds. `rl/bc.py`: behaviour cloning + DAgger. `rl/es.py`: evolution strategy.
- `experiments/evaluate.py`: fly any policy on N seeds of the real task. `lander/seeds.py`: the seed ranges.
- `publish.py`: builds the site from the runs; `publish_loop.sh` every 15 minutes.
