#!/bin/bash
# chain: wait for the narration, build the film, put it on the site, then publish podcast episode 3
cd "$(dirname "$0")/.."
while ! grep -q NARRATION_DONE logs/film_narrate.log; do
  grep -q -i "error\|Traceback" logs/film_narrate.log && { echo "narration failed"; exit 1; }
  sleep 30
done
nice -n 5 .venv/bin/python video/build_film.py --resume || exit 1
.venv/bin/python experiments/journal.py "Film: how the lander learned (the bake-off explained)" "A 13-minute narrated film on the Model tab explains the pilot (16 inputs, a 16-128-128-4 network firing live), the bake-off (PPO from scratch vs PPO from a copy vs an evolution strategy, same budget, same 500 fresh flights), why the winner won, the five tests that guard the result (frozen task, same seed = same flight, no peeking at the hidden wind, separate seed ranges, the winner's curse) and where the winner still fails. New on every replay: the wind. Blue arrow = the true wind (hidden from the pilot), orange = the pilot's own estimate (input 16), and dust drifting with the gusts."
git add docs/video/bakeoff.mp4 docs/model.html video/build_film.py video/bakeoff/script.md video/after_narration.sh lander/viz.py
nice -n 15 .venv/bin/python publish.py --push
setsid nohup ~/.claude/skills/project-podcast/scripts/publish.sh 03 > podcast/publish_03.log 2>&1 < /dev/null &
echo CHAIN_DONE
