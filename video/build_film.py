"""Build the explainer film about the bake-off: real flights (with the wind drawn), the live network, the charts.

    python video/build_film.py [--only 3,4] [--preview]      # needs voice/film-bakeoff exported (per-scene MP3s)
Output: docs/video/bakeoff.mp4 (1280x720, 25 fps). Scenes match video/bakeoff/script.md (## sections) one to one.
"""
import argparse
import csv
import glob
import io
import json
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lander.env import GOAL_WORLD, Flight, make_env, boulder_spec, W, INPUT_NAMES
from lander.policy import load_policy, NumpyMLP
from lander.viz import Painter, font, true_wind, BLUE, ORANGE

FPS = 25
SW, SH = 1280, 720
BG = (14, 17, 22)
INK = (232, 232, 236)
MUTED = (150, 152, 160)
ACC = (120, 160, 255)
GOOD = (105, 219, 124)
BAD = (255, 120, 120)
WARN = (255, 169, 77)
VOICE = 'voice/film-bakeoff'
OUT = 'docs/video'
TMP = 'video/build'
A_FINAL = 'runs/bake_A_ppo_scratch/final.json'
A_DIR = 'runs/bake_A_ppo_scratch/champions'
GAP = 0.7                       # silence after each scene


# ---------------- drawing helpers ----------------

def canvas():
    return Image.new('RGB', (SW, SH), BG)


def wrap(d, text, f, width):
    words, lines, cur = text.split(), [], ''
    for w in words:
        t = (cur + ' ' + w).strip()
        if d.textlength(t, font=f) > width and cur:
            lines.append(cur); cur = w
        else:
            cur = t
    if cur:
        lines.append(cur)
    return lines


def text(d, xy, s, size=28, fill=INK, bold=False, width=None, spacing=1.3):
    f = font(size, bold)
    x, y = xy
    lines = wrap(d, s, f, width) if width else [s]
    for l in lines:
        d.text((x, y), l, fill=fill, font=f)
        y += int(size * spacing)
    return y


def header(d, title, sub=None):
    text(d, (48, 30), title, 34, INK, bold=True)
    if sub:
        text(d, (48, 76), sub, 20, MUTED)


def fig_to_img(fig, w, h):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB').resize((w, h), Image.LANCZOS)


def dark_fig(w, h):
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor('#0e1116'); ax.set_facecolor('#0e1116')
    for s in ax.spines.values():
        s.set_color('#555a66')
    ax.tick_params(colors='#b0b3bb', labelsize=11)
    ax.xaxis.label.set_color('#d0d2d8'); ax.yaxis.label.set_color('#d0d2d8')
    ax.grid(color='#2a2e38', lw=0.8)
    return fig, ax


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


# ---------------- flights ----------------

def fly(policy, seed, world=GOAL_WORLD, every=2, extra=None, stochastic=False):
    """Fly and record frames (PIL 600x400 with wind + boulder) and per-frame extras. Returns frames, infos, result."""
    pol = load_policy(policy) if isinstance(policy, str) else policy
    env = make_env('rgb_array')
    f = Flight(env, seed, world)
    painter = Painter(seed)
    frames, infos = [], []
    rng = np.random.default_rng(seed)
    done = False
    while True:
        v = f.inputs()
        if f.steps % every == 0 or done:
            frames.append(painter.paint(env, f))
            infos.append(extra(v, f) if extra else None)
        if done:
            break
        if stochastic:
            z = pol.logits(v[None])[0]; p = np.exp(z - z.max()); p /= p.sum()
            a = int(rng.choice(4, p=p))
        else:
            a = pol(v)
        done = f.step(a)
    env.close()
    return frames, infos, f.result()


def outcome_label(res):
    if res['landed']:
        return 'LANDED between the flags', GOOD
    if res['rested']:
        return 'FAILED: came to rest outside the flags', BAD
    return {'crashed': ('FAILED: crashed', BAD), 'boulder': ('FAILED: touched the boulder', BAD),
            'timeout': ('FAILED: timeout (still flying after 1000 steps)', WARN)}[res['outcome']]


def clip_frame(frames, res, i, n_total, size=(900, 600), label=None):
    """frame i of a clip that should last n_total output frames: real time if it fits, faster if not; the last
    frame is held with the outcome."""
    speed = max(1.0, len(frames) / max(1, n_total - 20))
    k = int(i * speed)
    img = frames[min(k, len(frames) - 1)].resize(size, Image.LANCZOS)
    d = ImageDraw.Draw(img)
    if label:
        d.rectangle([0, size[1] - 34, size[0], size[1]], fill=(0, 0, 0))
        d.text((10, size[1] - 30), label, fill=INK, font=font(20))
    if k >= len(frames) - 1:
        s, c = outcome_label(res)
        fs = max(16, min(30, size[0] // 30))
        yb = int(size[1] * 0.80)
        d.rectangle([0, yb - fs, size[0], yb + fs], fill=(0, 0, 0))
        d.text((size[0] // 2 - d.textlength(s, font=font(fs, True)) / 2, yb - fs * 0.65), s, fill=c, font=font(fs, True))
    if speed > 1.05:
        d.text((size[0] - 70, 50), f'{speed:.1f}x', fill=MUTED, font=font(18))
    return img


def find_seed(policy, want, start=3_000_900_000, n=600):
    """First analysis seed where `policy` has the wanted outcome ('landed', 'offpad', 'crashed', 'boulder', 'timeout')."""
    from lander.env import run_episode
    pol = load_policy(policy)
    env = make_env()
    for s in range(start, start + n):
        r = run_episode(lambda v, f: pol(v, None), s, GOAL_WORLD, env=env)
        k = 'landed' if r['landed'] else ('offpad' if r['rested'] else r['outcome'])
        if k == want:
            return s
    raise RuntimeError(f'no {want} seed for {policy}')


# ---------------- scenes ----------------
# Each scene: function(n) -> generator of n PIL frames (1280x720).

def hold(img, n):
    for _ in range(n):
        yield img


def bullets_scene(title, sub, items, n, img=None, img_xy=None):
    """Bullets appear one after another over the first 80% of the scene."""
    for i in range(n):
        c = canvas(); d = ImageDraw.Draw(c)
        header(d, title, sub)
        y = 140
        shown = 1 + int(ease(i / (0.8 * n)) * (len(items) - 1) + 0.5) if items else 0
        for it in items[:shown]:
            d.ellipse([56, y + 11, 66, y + 21], fill=ACC)
            y = text(d, (84, y), it, 26, INK, width=(620 if img else 1120)) + 14
        if img:
            c.paste(img, img_xy)
        yield c


def flight_with_panel(frames, res, n, panel, label=None, title=None):
    """Flight on the left (750x500), panel(i_frame, draw, canvas) on the right."""
    speed = max(1.0, len(frames) / max(1, n - 20))
    for i in range(n):
        c = canvas(); d = ImageDraw.Draw(c)
        if title:
            header(d, title)
        k = min(int(i * speed), len(frames) - 1)
        c.paste(clip_frame(frames, res, i, n, (750, 500), label), (30, 130))
        panel(k, d, c)
        yield c


# 1 intro
def s_intro(n):
    seed = find_seed(A_FINAL, 'landed', 3_000_777_000)
    frames, _, res = fly(A_FINAL, seed)
    for i in range(n):
        c = canvas(); d = ImageDraw.Draw(c)
        c.paste(clip_frame(frames, res, i % (len(frames) + 40), len(frames) + 40, (900, 600)), (190, 100))
        t = min(1, i / 50)
        col = tuple(int(BG[j] + (INK[j] - BG[j]) * t) for j in range(3))
        text(d, (190, 22), 'How the lander learned', 40, col, bold=True)
        text(d, (190, 70), 'wind 15 · turbulence 1.5 · a boulder · goal: 98% of 1,000 unseen flights', 20, MUTED)
        if i > n * 0.55:
            d.rectangle([900, 640, 1270, 712], fill=(0, 0, 0))
            d.text((910, 646), 'blue arrow: the true wind (hidden)', fill=BLUE, font=font(17))
            d.text((910, 672), "orange: the pilot's own estimate", fill=ORANGE, font=font(17))
        yield c


# 2 outcomes
def s_outcomes(n):
    clips = []
    for pol, want, lab in [('teacher:v2', 'offpad', 'hand-written pilot v2: rests outside the flags'),
                           ('teacher:v2', 'crashed', 'hand-written pilot v2: a crash'),
                           ('teacher:v2', 'boulder', 'hand-written pilot v2: touches the boulder'),
                           ('models/bc_v2_128.json', 'timeout', 'a copied pilot: hovering until the timeout')]:
        s = find_seed(pol, want)
        fr, _, res = fly(pol, s)
        clips.append((fr, res, lab))
    parts = [0.33, 0.25, 0.2, 0.22]
    bounds = np.cumsum([0] + [int(p * n) for p in parts]); bounds[-1] = n
    for j, (fr, res, lab) in enumerate(clips):
        m = bounds[j + 1] - bounds[j]
        for i in range(m):
            c = canvas(); d = ImageDraw.Draw(c)
            header(d, 'What counts as a landing', 'only: at rest, between the flags. Four ways to fail:')
            c.paste(clip_frame(fr, res, i, m, (840, 560), lab), (40, 130))
            y = 150
            for k, (nm, col) in enumerate([('off-pad rest', BAD), ('crash', BAD), ('boulder touch', BAD), ('timeout', WARN)]):
                on = k == j
                d.rounded_rectangle([920, y, 1240, y + 90], 12, fill=(40, 44, 54) if on else (24, 27, 33),
                                    outline=col if on else (50, 54, 62), width=3 if on else 1)
                d.text((940, y + 28), nm, fill=INK if on else MUTED, font=font(26, on))
                y += 110
            yield c


# 3 senses
def s_senses(n):
    seed = find_seed(A_FINAL, 'landed', 3_000_778_000)
    frames, infos, res = fly(A_FINAL, seed, extra=lambda v, f: (np.array(v), true_wind(f.env)[0]))
    names = ['x', 'y', 'vx', 'vy', 'angle', 'spin', 'leg L', 'leg R', 'drift', '|x|', 'boulder dx', 'boulder dy',
             'boulder r', 'aim dx', 'above boulder', 'wind estimate']

    def panel(k, d, c):
        v, tw = infos[k]
        x0, y0 = 830, 135
        d.text((x0, y0 - 25), 'the 16 numbers it gets, this step', fill=MUTED, font=font(18))
        for j in range(16):
            y = y0 + 6 + j * 29
            col = ORANGE if j == 15 else (ACC if j < 8 else (170, 140, 255))
            d.text((x0, y), names[j], fill=INK if j == 15 else MUTED, font=font(17, j == 15))
            cx = x0 + 280
            val = float(np.clip(v[j], -1.5, 1.5))
            d.line([(cx, y + 2), (cx, y + 20)], fill=(70, 70, 80))
            d.rectangle([min(cx, cx + val * 100), y + 4, max(cx, cx + val * 100), y + 18], fill=col)
        y = y0 + 6 + 16 * 29 + 4
        d.text((x0, y), f'true wind (hidden from it): {tw:+.2f}', fill=BLUE, font=font(18))
        d.text((x0, y + 30), 'blue bars = game observations · purple = sensors', fill=MUTED, font=font(15))

    yield from flight_with_panel(frames, res, n, panel, title='What the pilot knows')


# 4 model
def s_model(n):
    pol = load_policy(A_FINAL)
    seed = find_seed(A_FINAL, 'landed', 3_000_779_000)

    def acts(v, f):
        h = np.clip((np.asarray(v) - pol.mean) / pol.std, -pol.clip, pol.clip)
        out = []
        for i, (Wm, b) in enumerate(pol.layers):
            h = h @ Wm.T + b
            if i < len(pol.layers) - 1:
                h = np.tanh(h)
            out.append(h.copy())
        return (np.asarray(v), out)
    frames, infos, res = fly(pol, seed, extra=acts)
    rng = np.random.default_rng(0)
    pick1, pick2 = np.sort(rng.choice(128, 18, replace=False)), np.sort(rng.choice(128, 18, replace=False))
    xs = [850, 960, 1070, 1180]
    ys_in = np.linspace(150, 640, 16); ys_h = np.linspace(160, 630, 18); ys_out = np.linspace(270, 520, 4)
    W1, W2, W3 = (np.asarray(L[0]) for L in pol.layers)
    base = Image.new('RGBA', (SW, SH), (0, 0, 0, 0)); bd = ImageDraw.Draw(base)
    def lines(xa, ya, xb, yb, Wm):
        m = np.abs(Wm).max()
        for i_b, y2 in enumerate(yb):
            for i_a, y1 in enumerate(ya):
                w = Wm[i_b, i_a] / m
                if abs(w) > 0.25:
                    col = (90, 140, 255, int(110 * abs(w))) if w > 0 else (255, 110, 110, int(110 * abs(w)))
                    bd.line([(xa, y1), (xb, y2)], fill=col, width=1)
    lines(xs[0], ys_in, xs[1], ys_h, W1[pick1][:, :])
    lines(xs[1], ys_h, xs[2], ys_h, W2[pick2][:, pick1])
    lines(xs[2], ys_h, xs[3], ys_out, W3[:, pick2])
    names = ['nothing', 'left', 'main', 'right']

    def panel(k, d, c):
        v, (h1, h2, z) = infos[k]
        c.paste(base, (0, 0), base)
        d.text((800, 118), '16 in      128       128      4 out', fill=MUTED, font=font(16))
        vin = np.clip((v - pol.mean) / pol.std, -2, 2) / 2
        for vals, x, ys in ((vin, xs[0], ys_in), (h1[pick1], xs[1], ys_h), (h2[pick2], xs[2], ys_h)):
            for val, y in zip(vals, ys):
                b = int(60 + 195 * abs(val))
                col = (b // 3, b // 2, b) if val >= 0 else (b, b // 3, b // 3)
                d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=col, outline=(90, 90, 100))
        a = int(np.argmax(z))
        for j, y in enumerate(ys_out):
            on = j == a
            d.ellipse([xs[3] - 13, y - 13, xs[3] + 13, y + 13], fill=GOOD if on else (40, 44, 54), outline=(120, 120, 130))
            d.text((xs[3] - 30, y + 16), names[j], fill=GOOD if on else MUTED, font=font(15, on))
        d.text((800, 668), 'red = negative · blue = positive · 18 of 128 hidden units drawn', fill=MUTED, font=font(14))

    for i, c in enumerate(flight_with_panel(frames, res, n, panel)):
        d = ImageDraw.Draw(c)
        header(d, 'The pilot is a neural network', '16 → 128 → 128 → 4 · about 19,000 weights · it fires the highest output')
        yield c


def results(name):
    return json.load(open(f'results/{name}.json'))


# 5 baselines
def s_baselines(n):
    items = [('pilot v1 (hand-written)', results('baseline_teacher_v1')), ('pilot v2 (tuned)', results('baseline_teacher_v2')),
             ('copy of v2 (NEAT seed)', results('baseline_clone_mlp16'))]
    for i in range(n):
        c = canvas(); d = ImageDraw.Draw(c)
        header(d, 'Where we started', 'each agent on the same 1,000 fresh flights')
        t = ease(i / (0.5 * n))
        x0, x1 = 420, 1180
        def X(p):
            return x0 + (p - 0.5) / 0.5 * (x1 - x0)
        for g in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            d.line([(X(g), 170), (X(g), 520)], fill=(40, 44, 54))
            d.text((X(g) - 18, 525), f'{int(g * 100)}%', fill=MUTED, font=font(16))
        for j, (nm, r) in enumerate(items):
            y = 200 + j * 100
            d.text((60, y + 12), nm, fill=INK, font=font(24))
            d.rectangle([x0, y, X(0.5 + (r['landed'] - 0.5) * t), y + 56], fill=ACC)
            if t > 0.95:
                d.text((X(r['landed']) + 10, y + 12), f"{100 * r['landed']:.1f}%", fill=INK, font=font(24, True))
                d.line([(X(r['ci'][0]), y + 28), (X(r['ci'][1]), y + 28)], fill=INK, width=2)
        d.line([(X(0.98), 170), (X(0.98), 520)], fill=BAD, width=3)
        d.text((X(0.98) - 60, 140), 'goal 98%', fill=BAD, font=font(20, True))
        if i > 0.6 * n:
            text(d, (60, 580), 'NEAT (last project) stalled at about 85%. 98% means removing almost 9 of every 10 remaining failures.',
                 24, WARN, width=1150)
        yield c


# 6 bake-off design
def s_bakeoff(n):
    lanes = [('A', 'PPO from a random network', 'reinforcement learning'), ('B', 'PPO from a copy of pilot v2', 'clone, then RL'),
             ('C', 'evolution strategy from the copy', 'perturb, keep the better')]
    for i in range(n):
        c = canvas(); d = ImageDraw.Draw(c)
        header(d, 'The bake-off', 'the same budget, the same exam')
        for j, (k, nm, sub) in enumerate(lanes):
            y = 160 + j * 130
            t = ease((i / n - 0.05 - j * 0.08) / 0.3)
            d.rounded_rectangle([60, y, 60 + int(560 * max(t, 0.02)), y + 100], 14, fill=(30, 36, 50), outline=ACC, width=2)
            if t > 0.5:
                d.text((84, y + 14), k, fill=ACC, font=font(40, True))
                d.text((140, y + 16), nm, fill=INK, font=font(24, True))
                d.text((140, y + 56), sub, fill=MUTED, font=font(19))
            if t > 0.9:
                d.text((660, y + 34), '45 min of all 8 CPU threads  →', fill=INK, font=font(22))
        t2 = ease((i / n - 0.45) / 0.2)
        if t2 > 0:
            d.rounded_rectangle([1010, 160, 1230, 520], 16, fill=(24, 40, 30), outline=GOOD, width=3)
            for k, l in enumerate(['the same', '500 flights', 'none of them', 'has seen', '', 'shared luck,', 'fair difference']):
                d.text((1030, 190 + k * 42), l, fill=GOOD if k < 4 else INK, font=font(22, k < 4))
        if i > 0.62 * n:
            text(d, (60, 580), 'Wind makes every flight different. If one candidate got easier test flights it would look better than it is. '
                 'Flying the same seeds shares the luck, so the difference that remains is real.', 23, INK, width=1160)
        yield c


# 7 PPO
def s_ppo(n):
    import torch
    from rl.ppo import AC
    blob = torch.load('runs/bake_A_ppo_scratch/checkpoint.pt', weights_only=False)
    net = AC((128, 128)); net.load_state_dict(blob['net'])
    pol = NumpyMLP(net.export())
    seed = find_seed(A_FINAL, 'landed', 3_000_780_000)

    def ex(v, f):
        with torch.no_grad():
            x = net.norm(torch.tensor(np.asarray(v), dtype=torch.float32)[None])
            p = torch.softmax(net.pi(x), -1)[0].numpy(); val = float(net.v(x)[0, 0])
        return p, val
    frames, infos, res = fly(pol, seed, extra=ex, stochastic=True)
    vals = [inf[1] for inf in infos]
    names = ['nothing', 'left', 'main', 'right']

    def panel(k, d, c):
        p, val = infos[k]
        x0 = 820
        d.text((x0, 140), 'the policy rolls the dice:', fill=INK, font=font(20, True))
        for j in range(4):
            y = 180 + j * 44
            d.text((x0, y + 4), names[j], fill=MUTED, font=font(18))
            d.rectangle([x0 + 100, y, x0 + 100 + int(300 * p[j]), y + 30], fill=ACC)
            d.text((x0 + 108 + int(300 * p[j]), y + 4), f'{100 * p[j]:.0f}%', fill=INK, font=font(17))
        d.text((x0, 380), "the critic's guess of how well", fill=INK, font=font(20, True))
        d.text((x0, 406), 'this flight will end:', fill=INK, font=font(20, True))
        gx, gy, gw, gh = x0, 450, 420, 170
        d.rectangle([gx, gy, gx + gw, gy + gh], outline=(60, 64, 74))
        lo, hi = min(vals) - 0.1, max(vals) + 0.1
        pts = [(gx + gw * i_ / max(1, len(vals) - 1), gy + gh - gh * (vals[i_] - lo) / (hi - lo)) for i_ in range(k + 1)]
        if len(pts) > 1:
            d.line(pts, fill=GOOD, width=2)
        d.text((gx, gy + gh + 6), 'the reward it still expects (drops after touchdown: the bonus is collected)', fill=MUTED, font=font(14))

    split = int(0.42 * n)
    yield from bullets_scene('PPO: reinforcement learning', 'learns from every step, not one number per flight', [
        'Starts from random weights: knows nothing.',
        'During training it outputs probabilities and rolls the dice, so it keeps trying new things.',
        'After each batch, every action is judged: better or worse than expected?',
        'A second network, the critic, learns what to expect.',
        'Better → a little more likely. Worse → a little less. A clip keeps each step small.'], split)
    yield from flight_with_panel(frames, res, n - split, panel, title='PPO during training: one stochastic flight')


def read_stats(run):
    return list(csv.DictReader(open(f'runs/{run}/stats.csv')))


def curve_img(runs, key, w, h, upto=None, ylab='% landed (200 fresh flights)', ylim=None, xkey='steps', goal=True, labels=None,
              colors=None):
    fig, ax = dark_fig(w, h)
    for j, run in enumerate(runs):
        rows = [r for r in read_stats(run) if r.get(key, '') != '']
        x = np.array([float(r[xkey]) for r in rows]) / (1e6 if xkey == 'steps' else 1)
        y = np.array([float(r[key]) for r in rows]) * (100 if key != 'entropy' else 1)
        if upto is not None:
            m = x <= upto[j] if isinstance(upto, list) else x <= upto
            x, y = x[m], y[m]
        col = (colors or ['#ffa94d', '#8ea6ff', '#69db7c'])[j]
        ax.plot(x, y, 'o' if key != 'entropy' else '-', ms=2.5, color=col, alpha=0.6, label=(labels or [run])[j])
        if key != 'entropy' and len(y) > 8:
            k = 9
            sm = np.convolve(y, np.ones(k) / k, mode='valid')
            ax.plot(x[k // 2:len(sm) + k // 2], sm, '-', color=col, lw=2.2)
    if goal and key != 'entropy':
        ax.axhline(98, color='#ff8787', ls='--', lw=1.2)
        ax.text(0.3, 98.6, 'goal 98%', color='#ff8787', fontsize=11)
    ax.set_xlabel('million steps of experience' if xkey == 'steps' else 'generation')
    ax.set_ylabel(ylab)
    if ylim:
        ax.set_ylim(*ylim)
    if labels:
        ax.legend(facecolor='#0e1116', edgecolor='#555a66', labelcolor='#d0d2d8', fontsize=11, loc='lower right')
    fig.tight_layout()
    return fig_to_img(fig, w, h)


# 8 learning
def s_learning(n):
    snaps = [('i00000_s0_goal.json', 'start (random)'), ('i00100_s0_goal.json', '1.6 M steps'),
             ('i00500_s0_goal.json', '8.2 M steps'), ('i01260_s0_goal.json', '20.6 M steps')]
    seed = 3_000_781_234
    clips = [fly(os.path.join(A_DIR, f), seed)[0::2] for f, _ in snaps] + [fly(A_FINAL, seed)[0::2]]
    labels = [l for _, l in snaps] + ['42 M steps (end)']
    split = int(0.6 * n)
    maxlen = max(len(c[0]) for c in clips)
    for i in range(split):
        c = canvas(); d = ImageDraw.Draw(c)
        header(d, 'The same network as it learned, on the same flight')
        for j, (fr, res) in enumerate(clips):
            x = 30 + (j % 3) * 410; y = 110 + (j // 3) * 300
            c.paste(clip_frame(fr, res, i, split, (400, 267), labels[j]), (x, y))
        yield c
    full = curve_img(['bake_A_ppo_scratch'], 'goal_landed', 1180, 560, ylim=(0, 100))
    for i in range(n - split):
        c = canvas(); d = ImageDraw.Draw(c)
        header(d, 'Learning curve of the winner', 'each dot: 200 fresh real-task flights, always the most likely action')
        t = ease(i / (0.6 * (n - split)))
        crop = full.crop((0, 0, int(80 + (1180 - 80) * t), 560))
        c.paste(crop, (50, 130))
        yield c


# 9 clone
def s_clone(n):
    split = int(0.5 * n)
    yield from bullets_scene('Candidate B: copy the pilot, then PPO', None, [
        'Behaviour cloning: the pilot flies; the network learns to predict its action at every step.',
        'DAgger: the copy flies itself, and the pilot says what it would have done there: it learns to recover from its own mistakes.',
        'The 2×128 copy lands 83.5% vs the pilot\'s 85.9% on the same 1,000 flights: a faithful copy.',
        'Then PPO took over... and it barely moved: 87.0% → 87.8% on the 500 bake-off flights.'], split)
    img = curve_img(['bake_A_ppo_scratch', 'bake_B_ppo_warm'], 'entropy', 1180, 520, ylab='exploration (policy entropy)',
                    labels=['A: from scratch', 'B: from the copy'], colors=['#ffa94d', '#8ea6ff'])
    for i in range(n - split):
        c = canvas(); d = ImageDraw.Draw(c)
        header(d, 'Why B stalled: it hardly explores', 'entropy: how much the pilot varies its actions (0 = always the same)')
        c.paste(img, (50, 130))
        if i > 0.4 * (n - split):
            text(d, (60, 660), 'A pilot that never tries anything new gives reinforcement learning nothing to learn from.', 22, WARN)
        yield c


# 10 es
def s_es(n):
    rows = read_stats('bake_C_es_warm')
    fig, ax = dark_fig(1180, 470)
    g = np.array([int(r['gen']) for r in rows]); pl = np.array([float(r['pop_landed']) for r in rows]) * 100
    ax.plot(g, pl, '-', color='#8ea6ff', lw=1.5, label='average of the 40 versions, 32 training flights')
    ck = [(int(r['gen']), float(r['check_landed']) * 100) for r in rows if r['check_landed']]
    ax.plot(*zip(*ck), 'o-', color='#ffa94d', label='the mean network, 200 fresh flights')
    ax.set_xlabel('generation'); ax.set_ylabel('% landed'); ax.set_ylim(70, 100)
    ax.legend(facecolor='#0e1116', edgecolor='#555a66', labelcolor='#d0d2d8', fontsize=11, loc='lower right')
    fig.tight_layout()
    img = fig_to_img(fig, 1180, 470)
    split = int(0.45 * n)
    yield from bullets_scene('Candidate C: an evolution strategy', None, [
        'Make 40 slightly changed versions of the copy.',
        'Each flies the same 32 flights.',
        'Move the weights toward the versions that landed more often.',
        'Only one number per flight per version, and 32 windy flights is a noisy measurement.'], split)
    for i in range(n - split):
        c = canvas(); d = ImageDraw.Draw(c)
        header(d, 'Candidate C over 45 minutes', 'it ended slightly below its start (87.0% → 83.4% on the bake-off flights)')
        c.paste(img, (50, 140))
        yield c


# 11 results
def s_results(n):
    items = [('A: PPO from scratch', results('bake_bake_A_ppo_scratch'), GOOD),
             ('B: PPO from the copy', results('bake_bake_B_ppo_warm'), ACC),
             ('C: evolution strategy', results('bake_bake_C_es_warm'), ACC),
             ('the copy itself (start of B, C)', results('bake_start_bc'), MUTED)]
    x0, x1 = 470, 1200
    X = lambda p: x0 + (p - 0.75) / 0.25 * (x1 - x0)
    for i in range(n):
        c = canvas(); d = ImageDraw.Draw(c)
        header(d, 'Results: the same 500 fresh flights', 'bars: landed · thin line: 95% interval (where the true rate most likely is)')
        t = ease(i / (0.35 * n))
        for g in (0.75, 0.8, 0.85, 0.9, 0.95, 1.0):
            d.line([(X(g), 150), (X(g), 560)], fill=(40, 44, 54))
            d.text((X(g) - 18, 565), f'{int(g * 100)}%', fill=MUTED, font=font(16))
        for j, (nm, r, col) in enumerate(items):
            y = 170 + j * 95
            d.text((50, y + 14), nm, fill=INK, font=font(23, j == 0))
            d.rectangle([x0, y, X(0.75 + (r['landed'] - 0.75) * t), y + 56], fill=col)
            if t > 0.95:
                d.line([(X(r['ci'][0]), y + 28), (X(r['ci'][1]), y + 28)], fill=INK, width=3)
                for e in r['ci']:
                    d.line([(X(e), y + 16), (X(e), y + 40)], fill=INK, width=3)
                d.text((x0 + 8, y + 14), f"{100 * r['landed']:.1f}%", fill=BG, font=font(24, True))
        d.line([(X(0.98), 150), (X(0.98), 560)], fill=BAD, width=3)
        if i > 0.55 * n:
            A = items[0][1]['outcomes']
            text(d, (50, 610), f"A's failures in 500 flights: boulder {A.get('boulder', 0)}, crash {A.get('crashed', 0)}, "
                 f"timeout {A.get('timeout', 0)}, off-pad {A.get('off-pad', 0)}. The pilot's most common failure is gone.", 23, GOOD, width=1180)
        yield c


# 12 tests
def s_tests(n):
    import hashlib
    parts = [0.26, 0.27, 0.22, 0.25]
    bounds = np.cumsum([0] + [int(p * n) for p in parts]); bounds[-1] = n
    # 1 frozen
    m = bounds[1]
    hashes = [(p, hashlib.sha256(open(p, 'rb').read()).hexdigest()) for p in ('lander/env.py', 'lander/route_params.json', 'lander/teacher.py')]
    for i in range(m):
        c = canvas(); d = ImageDraw.Draw(c)
        header(d, 'Test 1: the task is frozen', 'the code that defines the world is copied unchanged, and its fingerprint is checked')
        for j, (p, h) in enumerate(hashes):
            if i > j * m * 0.15:
                d.text((70, 190 + j * 90), p, fill=INK, font=font(26, True))
                d.text((70, 228 + j * 90), h, fill=GOOD, font=font(19))
                d.text((1150, 200 + j * 90), '✓', fill=GOOD, font=font(40, True))
        text(d, (70, 500), 'Change one character and the fingerprint changes, and the test fails. Otherwise we could reach 98% on an '
             'easier world without noticing.', 25, INK, width=1140)
        yield c
    # 2 determinism: the same seed twice, two separate simulators
    seed = 3_000_782_000
    f1, _, r1 = fly(A_FINAL, seed); f2, _, r2 = fly(A_FINAL, seed)
    same = r1['reward'] == r2['reward'] and r1['steps'] == r2['steps']
    m = bounds[2] - bounds[1]
    for i in range(m):
        c = canvas(); d = ImageDraw.Draw(c)
        header(d, 'Test 2: the same seed gives the same flight', f'seed {seed}, flown twice in two separate simulators')
        c.paste(clip_frame(f1, r1, i, m, (600, 400), 'simulator 1'), (30, 140))
        c.paste(clip_frame(f2, r2, i, m, (600, 400), 'simulator 2'), (650, 140))
        if i > 0.7 * m:
            text(d, (40, 570), f"score {r1['reward']:.4f} vs {r2['reward']:.4f}, {r1['steps']} vs {r2['steps']} steps: "
                 f"{'identical' if same else 'DIFFERENT'}. Without this, no comparison is fair.", 24, GOOD if same else BAD, width=1200)
        yield c
    # 3 no cheating
    m = bounds[3] - bounds[2]
    for i in range(m):
        c = canvas(); d = ImageDraw.Draw(c)
        header(d, 'Test 3: no peeking at the hidden wind')
        d.rounded_rectangle([60, 160, 500, 520], 16, fill=(30, 30, 38), outline=(90, 90, 100), width=2)
        d.text((90, 180), 'inside the simulator', fill=MUTED, font=font(24, True))
        for k, s in enumerate(['wind phase', 'turbulence phase', 'the seed', 'future gusts']):
            d.text((90, 240 + k * 60), s, fill=MUTED, font=font(24))
            if i > (0.15 + 0.1 * k) * m:
                d.line([(85, 255 + k * 60), (340, 255 + k * 60)], fill=BAD, width=4)
        d.rounded_rectangle([620, 160, 1220, 520], 16, fill=(24, 40, 30), outline=GOOD, width=2)
        d.text((650, 180), 'what the pilot may use', fill=GOOD, font=font(24, True))
        text(d, (650, 240), 'the 8 observations, and sensors computed from what it has seen so far and its own past actions', 24,
             INK, width=540)
        if i > 0.5 * m:
            text(d, (60, 570), 'The test recomputes every input from the observations alone, and flies the pilot with no access to the '
                 'simulator. The flight comes out identical.', 24, GOOD, width=1160)
        yield c
    # 4 seed ranges
    m = n - bounds[3]
    ranges = [('training', 0, 0.5, ACC), ('copying demos', 0.5, 1.0, (150, 130, 255)), ('stage checks', 1.0, 1.5, WARN),
              ('choosing champions', 1.5, 2.0, (200, 160, 90)), ('GOAL TEST (untouched)', 2.0, 3.0, GOOD), ('analysis', 3.0, 4.0, MUTED)]
    for i in range(m):
        c = canvas(); d = ImageDraw.Draw(c)
        header(d, 'Test 4: separate seeds for separate jobs', 'every flight is defined by a seed number (billions)')
        X = lambda v: 80 + v / 4 * 1120
        for k, (nm, a, b, col) in enumerate(ranges):
            if i > k * m * 0.1:
                d.rectangle([X(a), 300, X(b) - 4, 360], fill=col)
                d.text((X(a), 370 + (k % 2) * 34), nm, fill=col, font=font(19, True))
                d.text((X(a), 270), f'{a:g}', fill=MUTED, font=font(16))
        text(d, (80, 480), 'A test makes sure the ranges never overlap. If the pilot trained on its own exam, the exam would mean nothing.',
             25, INK, width=1120)
        yield c


# 13 winner's curse
def s_curse(n):
    rows = [r for r in read_stats('bake_A_ppo_scratch') if r['goal_landed']]
    late = [float(r['goal_landed']) * 100 for r in rows if float(r['steps']) > 36e6]
    fig, ax = dark_fig(1180, 470)
    rng = np.random.default_rng(1)
    ax.scatter(rng.uniform(-0.25, 0.25, len(late)), late, color='#ffa94d', s=40, label='quick checks, 200 flights each (last 6 M steps)')
    ax.scatter([0], [max(late)], color='#ff8787', s=160, zorder=5, label=f'the best one: {max(late):.1f}%')
    ax.axhline(np.mean(late), color='#ffa94d', lw=1.5, ls=':')
    ax.text(0.3, np.mean(late) + 0.2, f'average {np.mean(late):.1f}%', color='#ffa94d', fontsize=12)
    r = results('bake_bake_A_ppo_scratch')
    ax.errorbar([1.2], [100 * r['landed']], yerr=[[100 * (r['landed'] - r['ci'][0])], [100 * (r['ci'][1] - r['landed'])]],
                fmt='s', color='#69db7c', ms=12, capsize=8, lw=2, label=f"500 fresh flights: {100 * r['landed']:.1f}%")
    ax.set_xlim(-0.6, 1.8); ax.set_xticks([]); ax.set_ylabel('% landed'); ax.set_ylim(85, 100)
    ax.legend(facecolor='#0e1116', edgecolor='#555a66', labelcolor='#d0d2d8', fontsize=11, loc='lower right')
    fig.tight_layout()
    img = fig_to_img(fig, 1180, 470)
    for i in range(n):
        c = canvas(); d = ImageDraw.Draw(c)
        header(d, "Test 5: the winner's curse", 'pick the best of many noisy measurements, and you pick the luckiest one')
        c.paste(img, (50, 130))
        if i > 0.6 * n:
            text(d, (60, 620), 'Rule: a claim only counts on flights never used to choose anything. The goal: 1,000 flights, twice.', 24, GOOD)
        yield c


# 14 failures
def s_failures(n):
    r = results('bake_bake_A_ppo_scratch')
    seed = next(s for s in r['failed_seeds'])
    fr, _, res = fly(A_FINAL, seed)
    fs = set(r['failed_seeds'])
    seeds = range(r['start'], r['start'] + r['n'])
    off = lambda s: abs(boulder_spec(s)[0] - W / 2)
    bins = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 2.5)]
    rates = [np.mean([s in fs for s in seeds if lo <= off(s) < hi]) * 100 for lo, hi in bins]
    fig, ax = dark_fig(560, 440)
    ax.bar(range(5), rates, color=['#69db7c', '#69db7c', '#ff8787', '#ff8787', '#ff8787'])
    ax.set_xticks(range(5)); ax.set_xticklabels([f'{a:g}–{b:g}' for a, b in bins])
    ax.set_xlabel('boulder distance from the middle (m)'); ax.set_ylabel('% of flights failed')
    fig.tight_layout()
    img = fig_to_img(fig, 560, 440)
    for i in range(n):
        c = canvas(); d = ImageDraw.Draw(c)
        header(d, 'Where the winner still fails', 'mostly the boulder: about 6-7 flights in 100')
        c.paste(clip_frame(fr, res, i, int(0.6 * n), (660, 440), 'the winner, one of its failures'), (30, 150))
        if i > 0.45 * n:
            c.paste(img, (700, 150))
        yield c


# 15 next
def s_next(n):
    rows = [r for r in read_stats('M1') if r['goal_landed']]
    stages = [('1', 'real task, keep learning', 'it was still improving'), ('2', 'hard cases', '30% of flights repeat recent failures'),
              ('3', 'fine-tune', 'smaller steps, less exploration')]
    fig, ax = dark_fig(1180, 300)
    x = np.array([float(r['steps']) for r in rows]) / 1e6; y = np.array([float(r['goal_landed']) for r in rows]) * 100
    ax.plot(x, y, 'o', ms=3, color='#ffa94d', alpha=0.6)
    if len(y) > 9:
        sm = np.convolve(y, np.ones(9) / 9, mode='valid'); ax.plot(x[4:len(sm) + 4], sm, color='#ffa94d', lw=2.2)
    ax.axhline(98, color='#ff8787', ls='--'); ax.set_ylim(80, 100)
    ax.set_xlabel('million extra steps in the main run'); ax.set_ylabel('% landed (200 fresh)')
    fig.tight_layout()
    img = fig_to_img(fig, 1180, 300)
    for i in range(n):
        c = canvas(); d = ImageDraw.Draw(c)
        header(d, 'Next: the main run', 'the winner continues, through a three-stage curriculum')
        for j, (k, nm, sub) in enumerate(stages):
            x0 = 60 + j * 400
            d.rounded_rectangle([x0, 140, x0 + 370, 260], 14, fill=(30, 36, 50), outline=ACC if j == 0 else (60, 64, 74), width=2)
            d.text((x0 + 20, 152), k, fill=ACC, font=font(36, True))
            d.text((x0 + 70, 158), nm, fill=INK, font=font(22, True))
            text(d, (x0 + 70, 194), sub, 18, MUTED, width=290)
        if i > 0.35 * n:
            c.paste(img, (50, 290))
        if i > 0.75 * n:
            text(d, (60, 610), 'So far: about 92.5% on its checks. Too early to call a plateau. Site and podcast: jca374.github.io/lunarlander-98',
                 22, INK, width=1180)
        yield c


SCENES = [s_intro, s_outcomes, s_senses, s_model, s_baselines, s_bakeoff, s_ppo, s_learning, s_clone, s_es, s_results,
          s_tests, s_curse, s_failures, s_next]


def audio_path(k):
    p = os.path.join(VOICE, 'audio', 'voiceover', f'ch{k:02d}.mp3')
    return p if os.path.exists(p) else None


def duration(path):
    out = subprocess.run(['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', path],
                         capture_output=True, text=True).stdout
    return float(out)


def render_scene(k, fn, preview=False):
    a = audio_path(k)
    dur = (duration(a) if a else 20.0) + GAP
    n = int(dur * FPS)
    out = os.path.join(TMP, f'scene{k:02d}.mp4')
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{SW}x{SH}', '-r', str(FPS), '-i', '-']
    if a:
        cmd += ['-i', a, '-af', f'apad=pad_dur={GAP + 1}', '-c:a', 'aac', '-b:a', '128k', '-ar', '44100', '-ac', '1']
    cmd += ['-t', f'{n / FPS:.3f}', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '24', '-pix_fmt', 'yuv420p', '-r', str(FPS), out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    count = 0
    for img in fn(n):
        if count >= n:
            break
        if preview and count % (n // 3 + 1) == 0:
            img.save(os.path.join(TMP, f'preview{k:02d}_{count}.png'))
        p.stdin.write(img.tobytes())
        count += 1
    last = img
    while count < n:
        p.stdin.write(last.tobytes()); count += 1
    p.stdin.close(); p.wait()
    print(f'scene {k}: {fn.__name__} {dur:.1f} s', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='')
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--resume', action='store_true', help='keep scene files that already exist')
    a = ap.parse_args()
    os.makedirs(TMP, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    only = {int(x) for x in a.only.split(',') if x}
    files = []
    for k, fn in enumerate(SCENES, 1):
        out = os.path.join(TMP, f'scene{k:02d}.mp4')
        if (not only or k in only) and not (a.resume and os.path.exists(out)):
            render_scene(k, fn, a.preview)
        files.append(out)
    if not only:
        lst = os.path.join(TMP, 'list.txt')
        open(lst, 'w').write(''.join(f"file '{os.path.abspath(f)}'\n" for f in files))
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy',
                        '-movflags', '+faststart', os.path.join(OUT, 'bakeoff.mp4')], check=True)
        print('film:', os.path.join(OUT, 'bakeoff.mp4'), f"{duration(os.path.join(OUT, 'bakeoff.mp4')):.0f} s")


if __name__ == '__main__':
    main()
