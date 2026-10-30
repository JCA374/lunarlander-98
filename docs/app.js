function countdown(deadline) {
  const tick = () => {
    const s = Math.max(0, (new Date(deadline) - new Date()) / 1000);
    const d = Math.floor(s / 86400), h = Math.floor(s % 86400 / 3600), m = Math.floor(s % 3600 / 60);
    $('countdown').textContent = s ? `⏱ ${d} d ${h} h ${m} min left until Wednesday 4 November, 17:00` : 'Deadline reached.';
  };
  tick(); setInterval(tick, 30000);
}
const NAMES = { baseline_teacher_v1: 'Hand-written pilot v1', baseline_teacher_v2: 'Hand-written pilot v2 (CMA-ES-tuned gains)',
  baseline_clone_mlp16: 'Old clone of pilot v2 (16→24→4), NEAT seed' };

async function main() {
  const [status, stats, log] = await Promise.all([get('status.json'), get('stats.json'), get('log.json')]);
  if (!status) { $('updated').textContent = 'No data yet.'; return; }
  countdown(status.deadline);
  $('updated').textContent = `Updated ${status.updated}.`;
  const rows = [];
  for (const [k, r] of Object.entries(status.goals || {})) rows.push([r.label || k, r, r.note || 'goal check']);
  for (const [k, r] of Object.entries(status.baselines || {})) rows.push([NAMES[k] || k, r, 'baseline, 1,000 seeds']);
  rows.sort((a, b) => b[1].landed - a[1].landed);
  $('goaltable').innerHTML = '<tr><th>Agent</th><th>Landed</th><th>95% interval</th><th>Off-pad</th><th>Crash</th><th>Boulder</th><th>Timeout</th><th>Flights</th></tr>' +
    rows.map(([n, r, note]) => `<tr><td>${esc(n)}<div class="muted small">${esc(note)}</div></td><td class="num"><b>${pct(r.landed)}</b></td>
      <td class="num">${pct(r.ci[0])} – ${pct(r.ci[1])}</td><td class="num">${pct(r.offpad)}</td><td class="num">${pct(r.crashed)}</td>
      <td class="num">${pct(r.boulder)}</td><td class="num">${pct(r.timeout)}</td><td class="num">${r.n}</td></tr>`).join('') +
    '<tr><td><b>Goal</b></td><td class="num"><b>98.0%</b></td><td colspan="6" class="muted">on 1,000 unseen flights, then again on a second 1,000</td></tr>';
  const runs = status.runs || [];
  const main = runs.find((r) => r.alive && !r.run.startsWith('bake')) || runs.find((r) => r.alive) || runs[runs.length - 1];
  if (main) {
    const l = main.last || {};
    $('cards').innerHTML = [
      ['Run', main.run, main.alive ? 'training now' : 'stopped'],
      ['Stage', l.stage || '–', `${main.passed.length} stage(s) passed`],
      ['Experience', l.steps ? (l.steps / 1e6).toFixed(1) + ' M steps' : (l.gen ? 'gen ' + l.gen : '–'), l.secs ? `${l.secs} s per iteration` : ''],
      ['Best real-task check', main.best_goal ? pct(main.best_goal.landed) : '–', main.best_goal ? `${main.best_goal.n} fresh flights (best of many: flatters)` : ''],
      ['Goal', '98%', '1,000 unseen flights, twice'],
    ].map(([k, v, s]) => `<div class="card"><div class="k">${k}</div><div class="v">${esc(v)}</div><div class="s">${esc(s)}</div></div>`).join('');
    chart((stats || {})[main.run] || [], main.run);
  }
  $('log').innerHTML = (log || []).map(eventHtml).join('') || '<p class="muted">Nothing yet.</p>';
}

function chart(rows, run) {
  if (!window.Chart || !rows.length) return;
  const xk = rows[0].steps !== undefined ? 'steps' : 'gen';
  const X = (r) => xk === 'steps' ? r.steps / 1e6 : +r.gen;
  const gk = rows[0].goal_landed !== undefined ? 'goal_landed' : 'check_landed';
  const tk = rows[0].train_goal_landed !== undefined ? 'train_goal_landed' : 'pop_landed';
  const pts = (k) => rows.filter((r) => r[k] !== '' && r[k] != null).map((r) => ({ x: X(r), y: 100 * r[k] }));
  const g = pts(gk), t = pts(tk);
  const ys = g.concat(t).map((p) => p.y);
  const yMin = ys.length ? Math.max(0, Math.floor((Math.min(...ys) - 2) / 5) * 5) : 0;
  const yMax = Math.min(100, Math.max(100, Math.ceil((Math.max(...ys) + 1) / 2) * 2));
  $('chartnote').textContent = `Run ${run}. x axis: ${xk === 'steps' ? 'million simulated steps of experience' : 'generation'}.`;
  new Chart($('chart'), { type: 'line', data: { datasets: [
    { label: 'own training flights landed', data: t, borderColor: cssv('--accent'), borderWidth: 1.5, pointRadius: 0, tension: .2 },
    { label: '200 fresh real-task flights', data: g, showLine: true, borderColor: cssv('--warn'), backgroundColor: cssv('--warn'), pointRadius: 2.5, borderWidth: 1 },
    { label: 'goal 98%', data: [{ x: X(rows[0]), y: 98 }, { x: X(rows[rows.length - 1]), y: 98 }], borderColor: cssv('--bad'), borderDash: [5, 4], borderWidth: 1, pointRadius: 0 },
  ] }, options: { animation: false, maintainAspectRatio: false, interaction: { mode: 'nearest', intersect: false },
    scales: { x: { type: 'linear', ticks: { color: cssv('--muted') }, grid: { color: cssv('--line') } },
      y: { min: yMin, max: yMax, title: { display: true, text: '% landed', color: cssv('--muted') }, ticks: { color: cssv('--muted') }, grid: { color: cssv('--line') } } },
    plugins: { legend: { labels: { color: cssv('--ink'), boxWidth: 12 } } } } });
}

function chk(c) { return c ? `${pct(c.landed)} landed of ${c.n}; off-pad ${pct(c.offpad)}, crash ${pct(c.crashed)}, boulder ${pct(c.boulder)}, timeout ${pct(c.timeout)}` : ''; }
function eventHtml(e) {
  let title = '', body = '';
  const at = e.steps ? ` at ${(e.steps / 1e6).toFixed(1)} M steps` : '';
  switch (e.kind) {
    case 'journal': title = e.title; body = e.text; break;
    case 'run_start': title = `Run ${e.run} started`; body = e.note; break;
    case 'stage_start': title = `Stage started: ${e.stage}`; body = e.expect ? 'Expected: ' + e.expect : ''; break;
    case 'goal_best': title = `Best real-task check so far${at}`; body = chk(e.check) + '\nA best-of-many check flatters (winner\'s curse); claims are re-measured on more seeds.'; break;
    case 'check_pass': title = `Check passed in ${e.stage}, confirming on a fresh block`; body = `First: ${chk(e.check)}\nConfirm: ${chk(e.confirm)}`; break;
    case 'stage_pass': title = `✅ Stage passed: ${e.stage}${at}`; body = `Check: ${chk(e.check)}\nConfirm: ${chk(e.confirm)}`; break;
    case 'stage_cap': title = `Stage ${e.stage} reached its cap; moving on`; break;
    case 'time_limit': title = `Time limit reached (${e.minutes} min)`; break;
    case 'crash': title = '⚠️ Training crashed and was restarted from its last checkpoint'; body = e.note; break;
    case 'override': title = `Manual change: ${e.what}`; body = e.reason; break;
    case 'curriculum_done': title = 'Curriculum finished'; break;
    case 'goal_check': title = `🎯 Goal check: ${pct(e.landed)} landed on ${e.n} unseen flights`; body = `${e.label || e.policy}. 95% interval ${pct(e.ci[0])} – ${pct(e.ci[1])}. ${e.note || ''}`; break;
    default: title = e.kind;
  }
  const img = e.replay ? `<img loading="lazy" src="${e.replay}" alt="replay of the agent at this moment">` : '';
  return `<div class="ev"><div class="body"><div class="t">${esc(e.t)}${e.run ? ' · run ' + esc(e.run) : ''}</div>
    <h4><span class="tag ${e.kind}">${e.kind.replace('_', ' ')}</span>${esc(title)}</h4><p>${esc(body)}</p></div>${img}</div>`;
}
main();
