# CONTINUE: Lander 98, result and how to resume

## Result (2026-10-31 02:20, deadline was Wed 2026-11-04 17:00)

**Goal met.** The agent `models/best/M2_i08100.json` (a 16→128→128→4 tanh MLP trained with PPO from random weights)
landed between the flags on the real task (wind 15, turbulence 1.5, boulder, reference gymnasium env, no help):

| Test block | Seeds | Landed | 95% Wilson interval | Failures |
|---|---|---|---|---|
| 1 | 2,000,000,000 + 0..999 | **98.8%** (988) | 97.9–99.3% | boulder 8, timeout 4 |
| 2 | 2,000,010,000 + 0..999 | **98.4%** (984) | 97.4–99.0% | boulder 9, crash 4, timeout 3 |
| both | 2,000 | 98.6% (1,972) | 98.0–99.0% | |

Chosen before the goal test on 3,000 selection flights (98.5%), never on the test seeds. Results: `models/best/goal_*.json`.
Off-pad landings: 0 of 2,000. Median touchdown speed 0.26 (faster than the hand-written pilot's 0.07, but they hold).

Earlier agents on 1,000 fresh flights: pilot v1 72.2%, pilot v2 84.0%, the old NEAT seed clone 84.6% (NEAT itself ~85%).

## How it got there (about 4.5 hours of training, 254 M simulated steps)

1. **Bake-off** (45 min of the whole CPU each, 500 shared analysis flights): PPO from scratch 91.4%, PPO warm-started from a
   behaviour clone 87.8% (barely moved: a confident copy hardly explores), OpenAI-ES from the clone 83.4%.
2. **M1** continued the winner: stage 1 real task (plateau ~92.7%, boulder hits ~5%); stage 2 **hard cases** (30% of
   flights re-fly real-task seeds failed on recent checks) → ~96.3%; stage 3 fine-tune (lr 1e-4) → 96.6% on 1,000
   selection flights.
3. **M2**: hard cases + fine-tune settings, one stage with no early stop, a snapshot every 100 iterations → checks
   ~98% after ~130 M steps; 8 snapshots × 1,000 selection flights, top 4 re-flown on 2,000 more, then the goal test.

## What worked / what didn't

- Worked: PPO from scratch (learns from every step; no inherited habits: it never rests off-pad); the 16 inputs with the
  clean wind sensor; the hard-case curriculum (the biggest single gain); checks on fresh seeds, picking champions on
  selection seeds, the goal on untouched test seeds.
- Didn't: warm starts from the clone (PPO and ES); stage passes on 200+200 checks (passed by luck after many tries);
  ending the run when the last stage passed (CPU idle 1 h); `git pull --rebase --autostash` in the publisher (silently
  reverted code for an hour; removed).
- Not needed in the end: the native fast env (training ran at ~18,000 steps/s in the reference env) and a memory (GRU).

## Resume / reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install "gymnasium[box2d]==0.29.1" "numpy<2" torch matplotlib pillow pytest
.venv/bin/python -m pytest -q tests                                     # 14 tests, incl. the frozen-task sha256
.venv/bin/python -m experiments.goal_check models/best/M2_i08100.json --block 1   # re-measure the goal
# continue training from the last M2 checkpoint (local only, not in git):
setsid nohup ./supervise.sh M2 curricula/M2.json --snapshot-every 100 > logs/supervise_M2.log 2>&1 &
# or a fresh run like the winner: bake-off candidate A, then M1, then M2 (see curricula/*.json, --resume-from)
nice .venv/bin/python publish.py --push                                 # site update; ./publish_loop.sh every 15 min
```

## Open ideas

- The remaining ~1.4% failures: boulder touches (~0.85%), timeouts (~0.35%), crashes (~0.2%). A paired failure analysis
  (experiments/failures.py, boulder_map.py) on the champion would show whether they are near-unwinnable layouts (~0.65%
  per the old crude escape test) or fixable.
- Softer touchdowns (speed 0.26) for a margin; a longer M2 with lower lr; averaging snapshot weights.
