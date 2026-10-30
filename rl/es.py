"""OpenAI-style evolution strategy over the actor's weights, warm-started from a numpy policy.

    python -m rl.es --init models/bc_v2_128.json --run E1 [--pop 40 --flights 32 --sigma 0.01 --lr 0.01]

Every generation: `pop` antithetic pairs of weight perturbations (+eps, -eps), all flying the same `flights` training
seeds (common random numbers). Fitness per flight = landed (1/0) + 0.002 x game score: pays for landings first.
The mean moves along the rank-weighted perturbations (Adam). Every `check_every` generations the mean flies 200
fresh CHECK seeds of the real task.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')     # one BLAS thread per process: 8 workers x 8 threads overloaded the CPU (load 34)
import argparse
import csv
import json
import os
import signal
import time
from multiprocessing import Pool

import numpy as np

from lander.env import GOAL_WORLD, run_episode, make_env
from lander.policy import NumpyMLP
from lander.seeds import TRAIN, CHECK

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_W = {}


def _init(template):
    _W['t'] = template
    _W['env'] = make_env()


def flat(d):
    return np.concatenate([np.ravel(W) for L in d['layers'] for W in L])


def unflat(template, x):
    out, i = dict(template), 0
    layers = []
    for W, b in template['layers']:
        W, b = np.asarray(W), np.asarray(b)
        nW = x[i:i + W.size].reshape(W.shape); i += W.size
        nb = x[i:i + b.size]; i += b.size
        layers.append([nW, nb])
    out['layers'] = layers
    return out


def _fly(args):
    x, seeds = args
    pol = NumpyMLP(unflat(_W['t'], x))
    rs = [run_episode(pol, s, GOAL_WORLD, env=_W['env']) for s in seeds]
    return [r['landed'] + 0.002 * r['reward'] for r in rs], [r['landed'] for r in rs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--pop', type=int, default=20, help='antithetic pairs')
    ap.add_argument('--flights', type=int, default=32)
    ap.add_argument('--sigma', type=float, default=0.01)
    ap.add_argument('--lr', type=float, default=0.005)
    ap.add_argument('--check-every', type=int, default=10)
    ap.add_argument('--max-minutes', type=float, default=0)
    a = ap.parse_args()
    d = json.load(open(a.init))
    rdir = os.path.join(ROOT, 'runs', a.run)
    os.makedirs(os.path.join(rdir, 'champions'), exist_ok=True)
    stop = {'v': False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))
    ck = os.path.join(rdir, 'checkpoint.npz')
    x = flat(d); m = np.zeros_like(x); v = np.zeros_like(x); gen, n_check = 0, 0
    if os.path.exists(ck):
        z = np.load(ck)
        x, m, v, gen, n_check = z['x'], z['m'], z['v'], int(z['gen']), int(z['n_check'])
    rng = np.random.default_rng(1234 + gen)
    t_start = time.time()
    with Pool(8, initializer=_init, initargs=(d,)) as pool:
        while not stop['v'] and not (a.max_minutes and time.time() - t_start > a.max_minutes * 60):
            t0 = time.time()
            seeds = [TRAIN + 100_000_000 + gen * 1000 + i for i in range(a.flights)]
            eps = rng.standard_normal((a.pop, x.size))
            cands = [x + a.sigma * e for e in eps] + [x - a.sigma * e for e in eps]
            out = pool.map(_fly, [(c, seeds) for c in cands], chunksize=1)
            f = np.array([np.mean(o[0]) for o in out]); land = np.array([np.mean(o[1]) for o in out])
            ranks = np.empty(len(f)); ranks[np.argsort(f)] = np.arange(len(f))
            u = ranks / (len(f) - 1) - 0.5
            g = -(u[:a.pop] - u[a.pop:]) @ eps / (2 * a.pop * a.sigma)     # minus: Adam below descends
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
            t = gen + 1
            x = x - a.lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            row = {'gen': gen, 'pop_landed': round(float(land.mean()), 3), 'best_landed': round(float(land.max()), 3),
                   'pop_fit': round(float(f.mean()), 4), 'check_landed': '', 'secs': 0}
            if gen % a.check_every == 0:
                cs = [CHECK + 500_000 + n_check * 1000 + i for i in range(200)]
                n_check += 1
                parts = [cs[i::8] for i in range(8)]
                res = pool.map(_fly, [(x, p) for p in parts])
                row['check_landed'] = round(float(np.mean([l for o in res for l in o[1]])), 3)
                pd = unflat(d, x)
                json.dump({**pd, 'layers': [[np.asarray(W).tolist(), np.asarray(b).tolist()] for W, b in pd['layers']]},
                          open(os.path.join(rdir, 'champions', f'g{gen:05d}.json'), 'w'))
            row['secs'] = round(time.time() - t0, 1)
            new = not os.path.exists(os.path.join(rdir, 'stats.csv'))
            with open(os.path.join(rdir, 'stats.csv'), 'a', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            print(' | '.join(f'{k} {v_}' for k, v_ in row.items()), flush=True)
            gen += 1
            np.savez(ck, x=x, m=m, v=v, gen=gen, n_check=n_check)
    pd = unflat(d, x)
    json.dump({**pd, 'layers': [[np.asarray(W).tolist(), np.asarray(b).tolist()] for W, b in pd['layers']]},
              open(os.path.join(rdir, 'final.json'), 'w'))


if __name__ == '__main__':
    main()
