#!/bin/bash
# The bake-off (brief section 3): three candidates, 45 minutes of the whole CPU each, then all on the same 500 analysis seeds.
cd "$(dirname "$0")"
.venv/bin/python -m rl.ppo --curriculum curricula/bake_scratch.json --run bake_A_ppo_scratch --max-minutes 45 > logs/bake_A.log 2>&1
.venv/bin/python -m rl.ppo --curriculum curricula/bake_warm.json --run bake_B_ppo_warm --init models/bc_v2_128.json \
  --lr 1e-4 --ent 0.003 --critic-warmup 15 --max-minutes 45 > logs/bake_B.log 2>&1
.venv/bin/python -m rl.es --init models/bc_v2_128.json --run bake_C_es_warm --max-minutes 45 > logs/bake_C.log 2>&1
for r in bake_A_ppo_scratch bake_B_ppo_warm bake_C_es_warm; do
  .venv/bin/python -m experiments.evaluate runs/$r/final.json --start 3000100000 --n 500 --out results/bake_$r.json
done
.venv/bin/python -m experiments.evaluate models/bc_v2_128.json --start 3000100000 --n 500 --out results/bake_start_bc.json
echo BAKEOFF DONE
