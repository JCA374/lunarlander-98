"""Build the site's data from the runs and push it (every 15 min via publish_loop.sh).

docs/data/status.json     the main run, the latest numbers, goal table, baselines, bake-off
docs/data/stats.json      per-iteration numbers for the charts (every run, thinned)
docs/data/log.json        newest-first log: journal notes, run events, goal checks, each with a replay GIF of the
                          agent *as it was then* (rendered once, when the event is first seen)
docs/data/curricula.json  every run's curriculum (stages, rewards, pass checks, expected outcomes)

    python publish.py            # build only
    python publish.py --push     # build, commit docs/ + run logs, push
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import subprocess
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(ROOT, 'docs')
DATA = os.path.join(DOCS, 'data')
REPLAYS = os.path.join(DOCS, 'replays')
DEADLINE = '2026-11-04T17:00:00+01:00'
REPLAY_KINDS = {'goal_best', 'stage_pass', 'goal_check', 'champion'}
PAGES = ['index.html', 'curriculum.html', 'senses.html', 'model.html', 'speed.html', 'lessons.html']
SCRIPTS = ['app.js', 'common.js', 'curriculum.js', 'speed.js', 'model.js', 'senses.js']


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def replay_gif(policy_path, seed, out):
    """One real-task flight of a saved policy as a GIF (every 2nd frame), with the boulder, wind dust drifting with the
    true wind, and a gauge: the true wind (blue, hidden from the pilot) vs the pilot's own estimate (orange)."""
    from lander.env import GOAL_WORLD, run_episode, make_env
    from lander.policy import load_policy
    from lander.viz import Painter
    pol = load_policy(policy_path)
    env = make_env('rgb_array')
    frames = []
    painter = Painter(seed)

    def rec(f):
        if f.steps % 2:
            return
        frames.append(painter.paint(env, f).resize((420, 280)).convert('P', palette=Image.ADAPTIVE, colors=48))

    from PIL import Image
    r = run_episode(lambda v, f: pol(v, None), seed, GOAL_WORLD, env=env, record=rec)
    env.close()
    frames += [frames[-1]] * 15
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=40, loop=0, optimize=True)
    return r


def thin(rows, n=600):
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    keep = {int(i * step) for i in range(n)} | {len(rows) - 1}
    # always keep rows with a check
    return [r for i, r in enumerate(rows) if i in keep or r.get('goal_landed') or r.get('check_landed')]


def build():
    from lander.seeds import ANALYSIS
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(REPLAYS, exist_ok=True)
    log, stats_all, curricula, runs = [], {}, {}, []
    for rdir in sorted(glob.glob(os.path.join(ROOT, 'runs', '*')), key=os.path.getmtime):
        name = os.path.basename(rdir)
        if name.startswith('test_') or not os.path.exists(os.path.join(rdir, 'stats.csv')):
            continue
        cpath = os.path.join(rdir, 'curriculum.json')
        cur = json.load(open(cpath)) if os.path.exists(cpath) else None
        if cur:
            curricula[name] = cur
        rows = read_csv(os.path.join(rdir, 'stats.csv'))
        stats_all[name] = thin(rows)
        events = read_jsonl(os.path.join(rdir, 'events.jsonl'))
        for i, e in enumerate(events):
            e = dict(e, run=name)
            if e['kind'] in REPLAY_KINDS and e.get('champion'):
                e['replay'] = replay_for(e['champion'], f'{name}_{i:04d}.gif', ANALYSIS + 777_000 + i)
            e.pop('args', None)
            log.append(e)
        alive = subprocess.run(['pgrep', '-f', f'^.venv/bin/python -m rl\\.\\w+ .*--run {name}( |$)'],
                               capture_output=True).returncode == 0
        best = max((e['check'] for e in events if e['kind'] == 'goal_best'), default=None, key=lambda c: c['landed'])
        runs.append({'run': name, 'note': (cur or {}).get('note', ''), 'alive': alive, 'last': rows[-1] if rows else {},
                     'n_stages': len((cur or {}).get('stages', [])), 'best_goal': best,
                     'passed': [e['stage'] for e in events if e['kind'] == 'stage_pass']})
    for e in read_jsonl(os.path.join(ROOT, 'experiments', 'journal.jsonl')):
        log.append(dict(e, kind='journal'))
    for g in sorted(glob.glob(os.path.join(ROOT, 'results', 'goal_*.json'))):
        r = json.load(open(g))
        e = {'kind': 'goal_check', **{k: v for k, v in r.items() if k != 'failed_seeds'}}
        if r.get('policy'):
            e['replay'] = replay_for(r['policy'], 'goal_' + os.path.basename(g).replace('.json', '.gif'), ANALYSIS + 778_000)
        log.append(e)
    log.sort(key=lambda e: e['t'], reverse=True)
    res = lambda pat: {os.path.basename(p)[:-5]: {k: v for k, v in json.load(open(p)).items() if k != 'failed_seeds'}
                       for p in sorted(glob.glob(os.path.join(ROOT, 'results', pat)))}
    status = {'updated': time.strftime('%Y-%m-%d %H:%M'), 'deadline': DEADLINE, 'goal_landed': 0.98, 'runs': runs,
              'baselines': res('baseline_*.json'), 'bakeoff': res('bake_*.json'), 'goals': res('goal_*.json')}
    for name, obj in [('status', status), ('stats', stats_all), ('log', log), ('curricula', curricula)]:
        json.dump(obj, open(os.path.join(DATA, f'{name}.json'), 'w'), separators=(',', ':'))


def replay_for(policy, gif, seed):
    path = os.path.join(REPLAYS, gif)
    src = os.path.join(ROOT, policy) if not policy.startswith('teacher:') else policy
    if not os.path.exists(path) and (policy.startswith('teacher:') or os.path.exists(src)):
        try:
            replay_gif(src, seed, path)
        except Exception as ex:          # a replay must never stop the publisher
            print('replay failed', gif, ex)
    return f'replays/{gif}' if os.path.exists(path) else None


def bust_caches():
    """Pages are cached for 10 minutes: version the scripts so a browser never runs an old one."""
    for page in PAGES:
        path = os.path.join(DOCS, page)
        if not os.path.exists(path):
            continue
        html = open(path).read()
        for js in SCRIPTS + ['style.css']:
            p = os.path.join(DOCS, js)
            if os.path.exists(p):
                v = hashlib.md5(open(p, 'rb').read()).hexdigest()[:8]
                html = re.sub(rf'(src|href)="{re.escape(js)}(\?v=\w+)?"', rf'\1="{js}?v={v}"', html)
        open(path, 'w').write(html)


def push():
    os.chdir(ROOT)
    for f in ['docs', 'experiments/journal.jsonl', 'results', 'curricula']:
        if os.path.exists(f):
            subprocess.run(['git', 'add', f])
    for f in glob.glob('runs/*/stats.csv') + glob.glob('runs/*/events.jsonl') + glob.glob('runs/*/curriculum.json') \
            + glob.glob('runs/*/champions/*_goal.json') + glob.glob('runs/*/champions/*_passed.json') + glob.glob('runs/*/final.json'):
        if '/test_' not in f:
            subprocess.run(['git', 'add', '-f', f])
    if subprocess.run(['git', 'diff', '--cached', '--quiet']).returncode == 0:
        return print('nothing to publish')
    subprocess.run(['git', 'commit', '-q', '-m', f'Site update {time.strftime("%Y-%m-%d %H:%M")}'], check=True)
    # No 'pull --rebase --autostash' before pushing: on 2026-10-30 21:34 its autostash failed to re-apply and silently
    # reverted uncommitted code (rl/ppo.py, supervise.sh) for an hour. Only this laptop pushes; if a push is ever
    # rejected, merge (works with a dirty tree, never stashes) and push again.
    if subprocess.run(['git', 'push', '-q'], check=False).returncode != 0:
        subprocess.run(['git', 'pull', '-q', '--no-rebase', '--no-edit'], check=False)
        subprocess.run(['git', 'push', '-q'], check=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--push', action='store_true')
    a = ap.parse_args()
    build()
    bust_caches()
    if a.push:
        push()
