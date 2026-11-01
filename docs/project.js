// Lander-specific formatters for the generic site (ml-project-site skill).
window.worldText = function worldText(w = {}, ramp = {}) {
  const p = [];
  const wind = ramp.wind ? `${ramp.wind[0]}→${ramp.wind[1]}` : (w.wind ?? 15);
  const turb = ramp.turbulence ? `${ramp.turbulence[0]}→${ramp.turbulence[1]}` : (w.turbulence ?? 1.5);
  if (w.steady) p.push(`steady push ${wind}`); else p.push(+wind === 0 ? 'calm' : `wind ${wind}`);
  if (!w.steady && +turb !== 0) p.push(`turbulence ${turb}`);
  if (w.boulder) p.push(ramp.boulder_size ? `boulder (size ${ramp.boulder_size[0]}→${ramp.boulder_size[1]})` : 'boulder');
  if (w.goal_mix) p.push(`${Math.round(100 * w.goal_mix)}% of flights are the real task (rehearsal)`);
  return p.join(' + ');
}
window.rewardText = function rewardText(r = {}) {
  const p = ['game score (its shaping, fuel cost, ±100 at the end)'];
  p.push(`+${r.landing_bonus ?? 200} for coming to rest between the flags`);
  p.push(`−${r.offpad_penalty ?? 100} for coming to rest outside them`);
  p.push(`−${r.timeout_penalty ?? 100} for a timeout`);
  if (r.boulder_penalty) p.push(`−${r.boulder_penalty} extra for touching the boulder`);
  return p.join(', ');
}
window.passText = function passText(p = {}) {
  const q = [];
  if (p.landed != null) q.push(p.landed > 1 ? 'none (runs to its cap)' : `≥ ${pct(p.landed, 0)} landed on 200 fresh flights, confirmed on another 200`);
  if (p.offpad != null) q.push(`off-pad ≤ ${pct(p.offpad, 0)}`);
  if (p.timeout != null) q.push(`timeouts ≤ ${pct(p.timeout, 0)}`);
  return q.join(', ');
}

window.eventText = (e) => {
  const at = e.steps ? ` at ${(e.steps / 1e6).toFixed(1)} M steps` : '';
  if (e.kind === 'goal_best') return [`Best real-task check so far${at}`, `${pct(e.check.landed)} landed of ${e.check.n}; off-pad ${pct(e.check.offpad)}, crash ${pct(e.check.crashed)}, boulder ${pct(e.check.boulder)}, timeout ${pct(e.check.timeout)}\nA best-of-many check flatters (winner's curse); claims are re-measured on more seeds.`];
  if (e.kind === 'time_limit') return [`Time limit reached (${e.minutes} min)`, ''];
  return null;
};
