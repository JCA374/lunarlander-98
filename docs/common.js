// Shared: the navigation bar, helpers.
const NAV = [['index.html', 'Log'], ['curriculum.html', 'Curriculum'], ['senses.html', 'How it sees'], ['model.html', 'The model'],
  ['speed.html', 'Speed'], ['lessons.html', 'Lessons'], ['podcast/index.html', 'Podcast']];
(function () {
  const here = location.pathname.split('/').pop() || 'index.html';
  const nav = document.createElement('nav'); nav.className = 'top';
  nav.innerHTML = '<a class="brand" href="index.html">🌙 Lander 98</a>' + NAV.map(([h, t]) =>
    `<a href="${h}" class="${h === here ? 'on' : ''}">${t}</a>`).join('') + '<a href="https://github.com/JCA374/lunarlander-98">Code</a>';
  document.body.prepend(nav);
})();
const $ = (id) => document.getElementById(id);
const pct = (x, d = 1) => (x === '' || x == null || isNaN(x)) ? '–' : (100 * x).toFixed(d) + '%';
const esc = (s) => String(s ?? '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const get = (f) => fetch('data/' + f + '?t=' + Date.now()).then((r) => r.ok ? r.json() : null).catch(() => null);
const cssv = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function worldText(w = {}, ramp = {}) {
  const p = [];
  const wind = ramp.wind ? `${ramp.wind[0]}→${ramp.wind[1]}` : (w.wind ?? 15);
  const turb = ramp.turbulence ? `${ramp.turbulence[0]}→${ramp.turbulence[1]}` : (w.turbulence ?? 1.5);
  if (w.steady) p.push(`steady push ${wind}`); else p.push(+wind === 0 ? 'calm' : `wind ${wind}`);
  if (!w.steady && +turb !== 0) p.push(`turbulence ${turb}`);
  if (w.boulder) p.push(ramp.boulder_size ? `boulder (size ${ramp.boulder_size[0]}→${ramp.boulder_size[1]})` : 'boulder');
  if (w.goal_mix) p.push(`${Math.round(100 * w.goal_mix)}% of flights are the real task (rehearsal)`);
  return p.join(' + ');
}
function rewardText(r = {}) {
  const p = ['game score (its shaping, fuel cost, ±100 at the end)'];
  p.push(`+${r.landing_bonus ?? 200} for coming to rest between the flags`);
  p.push(`−${r.offpad_penalty ?? 100} for coming to rest outside them`);
  p.push(`−${r.timeout_penalty ?? 100} for a timeout`);
  if (r.boulder_penalty) p.push(`−${r.boulder_penalty} extra for touching the boulder`);
  return p.join(', ');
}
function passText(p = {}) {
  const q = [];
  if (p.landed != null) q.push(p.landed > 1 ? 'none (runs to its cap)' : `≥ ${pct(p.landed, 0)} landed on 200 fresh flights, confirmed on another 200`);
  if (p.offpad != null) q.push(`off-pad ≤ ${pct(p.offpad, 0)}`);
  if (p.timeout != null) q.push(`timeouts ≤ ${pct(p.timeout, 0)}`);
  return q.join(', ');
}
