"""The dashboard markup.

One self-contained page, no build step and no CDN: the inspector has to work on
an air-gapped campus network, and a debugging tool that needs the internet to
render is a debugging tool that will not load when you need it.
"""

from __future__ import annotations

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Campus pipeline inspector</title>
<style>
  :root {
    --bg: #10141a; --panel: #171d26; --line: #263041;
    --fg: #dde4ee; --dim: #7d8ba1; --accent: #58a6ff;
    --ok: #3cdc3c; --warn: #ffbe00; --bad: #ff4646; --idle: #96a0ae;
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  header { padding:10px 16px; border-bottom:1px solid var(--line);
           display:flex; gap:18px; align-items:baseline; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; font-weight:600; letter-spacing:.3px; }
  .k { color:var(--dim); }
  .k b { color:var(--fg); font-weight:600; }
  main { display:grid; grid-template-columns:1fr 420px; gap:14px; padding:14px;
         align-items:start; }
  @media (max-width:1100px){ main{ grid-template-columns:1fr; } }
  .cam { background:var(--panel); border:1px solid var(--line); border-radius:6px;
         margin-bottom:14px; overflow:hidden; }
  .cam > .hd { display:flex; gap:14px; padding:8px 12px; border-bottom:1px solid var(--line);
               flex-wrap:wrap; align-items:center; }
  .name { font-weight:600; }
  .dot { width:8px; height:8px; border-radius:50%; display:inline-block; }
  .live { background:var(--ok); } .stall { background:var(--warn); }
  .dead { background:var(--bad); } .idk { background:var(--idle); }
  img { display:block; width:100%; background:#000; }
  .faces { padding:4px 12px 12px; }
  .face { border-top:1px solid var(--line); padding:9px 0; }
  .face:first-child { border-top:0; }
  .fhead { display:flex; gap:10px; align-items:center; }
  .pill { padding:1px 7px; border-radius:9px; font-size:11px; border:1px solid; }
  .p-ok { color:var(--ok); border-color:var(--ok); }
  .p-tr { color:var(--warn); border-color:var(--warn); }
  .p-bad { color:var(--bad); border-color:var(--bad); }
  .p-idle { color:var(--idle); border-color:var(--idle); }
  .p-con { color:#00d7ff; border-color:#00d7ff; }
  .warn-line { margin:5px 0 2px; color:#00d7ff; font-size:11.5px; line-height:1.5; }
  .warn-line b { color:#00d7ff; }
  .who { font-weight:600; color:var(--ok); }
  .who.none { color:var(--idle); }
  .why { color:var(--bad); }
  .metrics { display:grid; grid-template-columns:repeat(auto-fill,minmax(74px,1fr));
             gap:3px 10px; margin:6px 0 2px; color:var(--dim); font-size:11px; }
  .metrics b { color:var(--fg); font-weight:500; }
  table { width:100%; border-collapse:collapse; margin-top:5px; font-size:11.5px; }
  th { text-align:left; color:var(--dim); font-weight:500; padding:1px 4px 1px 0; }
  td { padding:1px 4px 1px 0; }
  .bar { height:5px; background:#223; border-radius:3px; overflow:hidden;
         position:relative; margin-top:1px; }
  .bar i { position:absolute; inset:0 auto 0 0; background:var(--accent); display:block; }
  .bar.win i { background:var(--ok); }
  .runner { padding:1px 0; }
  .runner .sc { display:flex; gap:8px; }
  .runner .sc span { width:96px; }
  .runner .bar { flex:1; }
  aside { background:var(--panel); border:1px solid var(--line); border-radius:6px;
          padding:10px 12px; }
  aside h2 { font-size:12px; text-transform:uppercase; letter-spacing:.8px;
             color:var(--dim); margin:0 0 8px; font-weight:600; }
  .ev { border-top:1px solid var(--line); padding:7px 0; font-size:11.5px; }
  .ev:first-child { border-top:0; }
  .ev .t { color:var(--dim); }
  .note { color:var(--dim); font-size:11px; margin-top:10px; line-height:1.6; }
  .legend { display:flex; gap:12px; flex-wrap:wrap; color:var(--dim); font-size:11px; }
  .legend span::before { content:''; display:inline-block; width:8px; height:8px;
                         border-radius:2px; margin-right:4px; }
  .l-ok::before { background:var(--ok); } .l-tr::before { background:var(--warn); }
  .l-bad::before { background:var(--bad); } .l-idle::before { background:var(--idle); }
  .l-con::before { background:#00d7ff; }
  .rr { display:flex; justify-content:space-between; gap:10px; padding:2px 0;
        border-bottom:1px solid #1c2431; }
  .rr.live { color:var(--ok); }
  .rr.live b { color:var(--ok); }
  code { background:#0d1117; padding:1px 4px; border-radius:3px; color:var(--accent); }
</style>
</head>
<body>
<header>
  <h1>Campus pipeline inspector</h1>
  <div class="k">gallery <b id="gstat">-</b></div>
  <div class="k">detector <b id="dstat">-</b></div>
  <div class="k">embedder <b id="estat">-</b></div>
  <div class="k">updated <b id="upd">-</b></div>
</header>

<main>
  <div>
    <div id="cams"></div>
    <aside>
      <h2>Committed identities</h2>
      <div id="events"><span class="k">none yet</span></div>
      <div class="note">
        A commit only fires after the temporal verifier sees one identity hold a
        plurality of a sliding window <em>and</em> clear the runner-up by the
        margin threshold. A camera full of faces with no commits is the verifier
        declining to guess, which is the correct behaviour when scores are close
        &mdash; look at the margin column before concluding the system is broken.
      </div>
      <div class="legend" style="margin-top:10px">
        <span class="l-ok">resolved</span>
        <span class="l-tr">tracked, uncommitted</span>
        <span class="l-bad">resolved below score floor</span>
        <span class="l-con">contested &mdash; evidence disagrees</span>
        <span class="l-idle">rejected by quality gate</span>
      </div>
    </aside>
  </div>
  <aside>
    <h2>Enrolled roster</h2>
    <input id="q" placeholder="search student id..." autocomplete="off"
           style="width:100%;padding:5px 8px;background:#0d1117;color:var(--fg);
                  border:1px solid var(--line);border-radius:4px;font:inherit">
    <div class="k" id="rosterCount" style="margin:7px 0"></div>
    <div id="roster" style="max-height:46vh;overflow:auto"></div>
    <div class="note" id="rosterNote"></div>
  </aside>
  <aside>
    <h2>Thresholds in force</h2>
    <table id="thr"></table>
  </aside>
</main>

<script>
const META = __META__;
document.getElementById('gstat').textContent =
  META.gallery ? (META.gallery.students + ' students / ' + META.gallery.dim + '-d') : 'none';
document.getElementById('dstat').textContent = META.detector || '-';
document.getElementById('estat').textContent = META.embedder || '-';

document.getElementById('thr').innerHTML = Object.entries(META.quality || {})
  .map(([k,v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join('');

const esc = s => String(s == null ? '' : s)
  .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

function faceRow(f) {
  const pct = v => Math.max(0, Math.min(100, v * 100));
  let cls, tag;
  if (!f.passed) { cls = 'p-idle'; tag = 'rejected'; }
  else if (f.outcome === 'contested') { cls = 'p-con'; tag = 'CONTESTED'; }
  else if (f.student_id && f.score >= 0.38) { cls = 'p-ok'; tag = esc(f.student_id); }
  else if (f.student_id) { cls = 'p-bad'; tag = esc(f.student_id); }
  else { cls = 'p-tr'; tag = 'searching'; }
  const cands = (f.candidates || []).map(c => `
    <div class="runner">
      <div class="sc"><span>${esc(c.student_id)}</span>
        <b>${c.score.toFixed(3)}</b>
        <div class="bar ${c.rank===1?'win':''}"><i style="width:${pct(c.score)}%"></i></div>
      </div>
    </div>`).join('');
  const marginTxt = (f.candidates && f.candidates.length > 1)
    ? (f.candidates[0].score - f.candidates[1].score).toFixed(3) : '-';
  const ev = f.evidence || {};
  return `<div class="face">
    <div class="fhead">
      <span class="pill ${cls}">${tag}</span>
      <span class="k">now ${f.face_px}px det ${f.det_score.toFixed(2)}</span>
      ${f.searched_best_px ? `<span class="k">searched on <b>${f.searched_best_px}px</b> (best of ${ev.kept||1}/${ev.seen||1})</span>` : ''}
      ${f.margin ? `<span class="k">margin <b>${marginTxt}</b></span>` : ''}
    </div>
    ${f.outcome === 'contested' ? `<div class="warn-line">
        Evidence disagrees: the track is committed to <b>${esc(f.student_id)}</b>
        but the fresh frames point at <b>${esc(f.contested_by || 'nobody in particular')}</b>.
        Neither is asserted. This is what a subject who has turned away looks
        like &mdash; it is not a second person in frame.</div>` : ''}
    ${f.passed ? `<div class="metrics">
      <span>blur <b>${f.blur}</b></span><span>yaw <b>${f.yaw}&deg;</b></span>
      <span>pitch <b>${f.pitch}&deg;</b></span><span>roll <b>${f.roll}&deg;</b></span>
      <span>bright <b>${f.brightness}</b></span><span>contrast <b>${f.contrast}</b></span>
      <span>occl <b>${f.occlusion}</b></span></div>` : ''}
    ${!f.passed && f.reasons.length ? `<div class="why">${f.reasons.map(esc).join('; ')}</div>` : ''}
    ${cands}
  </div>`;
}

function camBlock(c) {
  const live = c.stream_state === 'live' ? 'live'
             : c.stream_state === 'stalled' ? 'stall' : 'dead';
  const st = c.stream_age_s == null ? 'connecting' :
             c.stream_age_s < 2 ? 'live' : c.stream_age_s < 10 ? 'stall' : 'dead';
  return `<div class="cam">
    <div class="hd">
      <span class="dot ${st}"></span>
      <span class="name">${esc(c.name || c.camera_id)}</span>
      <span class="k">${esc(c.camera_id)}</span>
      <span class="k">${c.resolution[0]}x${c.resolution[1]}${c.rotation ? ' rot'+c.rotation : ''}</span>
      <span class="k">zone <b>${esc(c.zone)}</b></span>
      <span class="k">panel <b>${c.fps} fps</b></span>
      <span class="k">analysis <b>${c.analysis_fps} fps</b></span>
      <span class="k">${c.tiles} tiles</span>
      <span class="k">det <b>${c.total_ms}ms</b></span>
      <span class="k">faces <b>${c.detected}</b> kept <b>${c.kept}</b></span>
      ${c.skipped_frames ? `<span class="k">skipped <b>${c.skipped_frames}</b></span>` : ''}
      ${c.stream_error ? `<span class="why">${esc(c.stream_error)}</span>` : ''}
    </div>
    <img src="/stream/${encodeURIComponent(c.camera_id)}" alt="${esc(c.camera_id)}">
    <div class="faces">${c.faces.length
        ? c.faces.map(faceRow).join('')
        : '<div class="k">no faces detected in the latest frame</div>'}</div>
  </div>`;
}

async function tick() {
  try {
    const s = await (await fetch('/api/state', {cache:'no-store'})).json();
    document.getElementById('cams').innerHTML = s.cameras.map(camBlock).join('');
    document.getElementById('events').innerHTML = s.events.length
      ? s.events.map(e => `<div class="ev">
          <span class="who">${esc(e.student_id)}</span>
          <span class="k"> @ ${esc(e.camera_id)}</span>
          <div class="t">score ${e.median_score.toFixed(3)} &middot; margin ${e.margin.toFixed(3)}
            &middot; ${e.evidence_frames} frames</div></div>`).join('')
      : '<span class="k">none yet</span>';
    document.getElementById('upd').textContent = new Date().toLocaleTimeString();
  } catch (e) {
    document.getElementById('upd').textContent = 'disconnected';
  }
}

// Which enrolled students are currently a top-1 candidate on any camera? Lets
// you answer "is he in the system right now" without reading every panel.
function seenSet() {
  const seen = new Set();
  for (const c of (window.__state?.cameras || []))
    for (const f of (c.faces || []))
      if (f.passed && f.candidates && f.candidates[0])
        seen.add(f.candidates[0].student_id);
  return seen;
}

let rosterTimer = null;
async function loadRoster() {
  const q = document.getElementById('q').value.trim();
  try {
    const r = await (await fetch('/api/gallery/students?q=' + encodeURIComponent(q),
                                   {cache:'no-store'})).json();
    const seen = seenSet();
    const needle = q.trim().toUpperCase();
    const rows = r.students.slice(0, needle ? 200 : 120);
    document.getElementById('rosterCount').textContent =
      r.query ? `${r.matched} of ${r.total} match "${r.query}"`
              : `${r.total} students enrolled (showing ${rows.length})`;
    document.getElementById('roster').innerHTML = rows.length
      ? rows.map(x => {
          const live = seen.has(x.student_id);
          return `<div class="rr ${live ? 'live' : ''}">
            <span>${esc(x.student_id)}</span>
            <span class="k">${x.photos} ph${live ? ' &middot; <b>on camera</b>' : ''}</span>
          </div>`;
        }).join('')
      : '<div class="k">no students match</div>';
    document.getElementById('rosterNote').innerHTML = needle
      ? 'No match means the id is not in the gallery. Enrol it with:<br>' +
        '<code>campus gallery --photos DIR --out models/gallery.npz --merge</code>'
      : 'Search an id to confirm an enrollment landed. A student only appears ' +
        'as <b>on camera</b> when they are currently a top-1 candidate.';
  } catch (e) {
    document.getElementById('roster').innerHTML =
      '<div class="k">roster unavailable (no gallery loaded)</div>';
  }
}

document.getElementById('q').addEventListener('input', () => {
  clearTimeout(rosterTimer);
  rosterTimer = setTimeout(loadRoster, 200);
});

const _origTick = tick;
async function tickWrapped() {
  await _origTick();
  try { window.__state = await (await fetch('/api/state', {cache:'no-store'})).json(); } catch(e){}
  await loadRoster();
}
tickWrapped();
setInterval(tickWrapped, 1500);
</script>
</body>
</html>
"""
