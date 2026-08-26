// Draws the measurements in results/noise_bench.json and nothing else.
//
// Every number on this page came out of a run that is committed to the
// repository. There is no model here and no inference: the page reads what the
// benchmark measured, which is the only way a results page can be checked.

const el = (id) => document.getElementById(id);
const plot = el('plot');
const ctx = plot.getContext('2d');
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

// The families the benchmark varies, in the order a reader should meet them.
// "clean" is prepended to every family so each curve starts from the baseline
// rather than from its own mildest noise.
const FAMILIES = [
  { key: 'ocr', label: 'OCR', match: (c) => c.startsWith('ocr') },
  { key: 'whitespace', label: 'Whitespace', match: (c) => c.startsWith('whitespace') },
  { key: 'hyphen', label: 'Hyphens', match: (c) => c.startsWith('hyphen') },
  { key: 'header', label: 'Headers', match: (c) => c.startsWith('header') },
  { key: 'redaction', label: 'Redaction', match: (c) => c.startsWith('redaction') },
  { key: 'scan', label: 'Whole scan', match: (c) => c.startsWith('scan') },
];

const state = { rows: [], family: 'ocr', retriever: 'hybrid', step: 0 };

const pct = (v) => (v === null || v === undefined ? '-' : v.toFixed(2));

function series() {
  const fam = FAMILIES.find((f) => f.key === state.family);
  const clean = state.rows.find((r) => r.condition === 'clean' && r.retriever === state.retriever);
  const rest = state.rows
    .filter((r) => r.retriever === state.retriever && fam.match(r.condition))
    .sort((a, b) => a.cer - b.cer);
  return clean ? [clean, ...rest] : rest;
}

function draw(points, active) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w0 = plot.clientWidth || 1200;
  // Capped: past a point a wider screen only adds empty plot, not resolution.
  const h0 = Math.min(Math.round(w0 * 0.38), 430);
  plot.width = Math.round(w0 * dpr);
  plot.height = Math.round(h0 * dpr);
  plot.style.height = h0 + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w0, h0);

  const pad = { l: 58, r: 22, t: 22, b: 52 };
  const w = w0 - pad.l - pad.r;
  const h = h0 - pad.t - pad.b;
  if (points.length < 2) return;

  const X = (i) => pad.l + (i / (points.length - 1)) * w;
  const Y = (v) => pad.t + h - v * h;

  // grid and axis
  ctx.strokeStyle = css('--hair');
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + h); ctx.lineTo(pad.l + w, pad.t + h);
  ctx.stroke();
  ctx.font = "11px 'Courier New', monospace";
  ctx.fillStyle = css('--faint');
  ctx.textAlign = 'right';
  for (let v = 0; v <= 1.0001; v += 0.25) {
    const y = Y(v);
    ctx.fillText(v.toFixed(2), pad.l - 8, y + 3);
    if (v > 0) {
      ctx.strokeStyle = '#e8e3d6';
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + w, y); ctx.stroke();
    }
  }

  // the band between what the dashboard sees and what survived
  ctx.beginPath();
  points.forEach((p, i) => (i ? ctx.lineTo(X(i), Y(p.recall_at_5)) : ctx.moveTo(X(i), Y(p.recall_at_5))));
  for (let i = points.length - 1; i >= 0; i--) ctx.lineTo(X(i), Y(points[i].ans_exact_at_5));
  ctx.closePath();
  ctx.globalAlpha = 0.13;
  ctx.fillStyle = css('--bad');
  ctx.fill();
  ctx.globalAlpha = 1;

  const line = (get, colour, dash, width) => {
    ctx.save();
    ctx.beginPath();
    points.forEach((p, i) => (i ? ctx.lineTo(X(i), Y(get(p))) : ctx.moveTo(X(i), Y(get(p)))));
    ctx.strokeStyle = colour;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.stroke();
    ctx.restore();
    points.forEach((p, i) => {
      ctx.beginPath();
      ctx.arc(X(i), Y(get(p)), i === active ? 4.5 : 2.5, 0, Math.PI * 2);
      ctx.fillStyle = colour;
      ctx.fill();
    });
  };

  // Solid for the metric that lies, dashed for the truth, dotted for the fix.
  // Never colour alone: this has to survive a greyscale print.
  line((p) => p.recall_at_5, css('--ok'), [], 1.8);
  line((p) => p.ans_exact_at_5, css('--bad'), [6, 4], 1.8);
  line((p) => p.ans_canon_at_5, css('--ox'), [2, 3], 1.4);

  // where the scrubber is
  if (active >= 0 && active < points.length) {
    ctx.save();
    ctx.strokeStyle = css('--ox');
    ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(X(active), pad.t); ctx.lineTo(X(active), pad.t + h); ctx.stroke();
    ctx.restore();
  }

  // x labels
  ctx.textAlign = 'center';
  ctx.fillStyle = css('--faint');
  points.forEach((p, i) => {
    const label = p.condition === 'clean' ? 'clean' : `${(p.cer * 100).toFixed(1)}%`;
    ctx.fillText(label, X(i), pad.t + h + 16);
  });
  ctx.fillText('character error rate', pad.l + w / 2, h0 - 8);

  // legend, inside the plot so it cannot be cropped away
  ctx.textAlign = 'left';
  ctx.font = "13px 'Times New Roman', serif";
  const key = [
    ['recall@5, what the dashboard plots', css('--ok'), []],
    ['answer still recoverable', css('--bad'), [6, 4]],
    ['after canonicalizing', css('--ox'), [2, 3]],
  ];
  // Bottom left: the curves live in the upper band, so a legend up there sits on
  // top of the very line it is labelling.
  const legendTop = pad.t + h - 18 - (key.length - 1) * 18;
  key.forEach(([text, colour, dash], i) => {
    const y = legendTop + i * 18;
    ctx.save();
    ctx.strokeStyle = colour;
    ctx.lineWidth = 1.8;
    ctx.setLineDash(dash);
    ctx.beginPath(); ctx.moveTo(pad.l + 12, y); ctx.lineTo(pad.l + 44, y); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = css('--sub');
    ctx.fillText(text, pad.l + 52, y + 4);
  });
}

function render() {
  const points = series();
  const scrub = el('scrub');
  scrub.max = String(Math.max(points.length - 1, 0));
  state.step = Math.min(state.step, points.length - 1);
  scrub.value = String(state.step);
  const p = points[state.step];
  draw(points, state.step);
  if (!p) return;

  el('r-cond').textContent = p.condition;
  el('r-cer').textContent = `${(p.cer * 100).toFixed(2)}%`;
  el('r-recall').textContent = pct(p.recall_at_5);
  el('r-exact').textContent = pct(p.ans_exact_at_5);
  el('r-canon').textContent = pct(p.ans_canon_at_5);

  const gap = p.recall_at_5 - p.ans_exact_at_5;
  const b = el('banner');
  if (gap >= 0.2) {
    b.className = 'banner alarm';
    b.textContent =
      `Retrieval reports ${pct(p.recall_at_5)} and ${Math.round((1 - p.ans_exact_at_5) * 100)}% of answers ` +
      `are no longer recoverable. A dashboard watching recall shows nothing wrong.`;
  } else if (gap > 0.05) {
    b.className = 'banner';
    b.textContent = `Starting to come apart: recall ${pct(p.recall_at_5)}, answers ${pct(p.ans_exact_at_5)}.`;
  } else {
    b.className = 'banner calm';
    b.textContent =
      p.condition === 'clean'
        ? 'Clean documents. The two metrics agree, which is why the disagreement goes unnoticed.'
        : `This noise costs almost nothing: recall ${pct(p.recall_at_5)}, answers ${pct(p.ans_exact_at_5)}.`;
  }
}

function buildTable() {
  const rows = state.rows
    .filter((r) => r.retriever === state.retriever && r.condition !== 'clean')
    .sort((a, b) => a.ans_exact_at_5 - b.ans_exact_at_5);
  const cells = rows
    .map((r) => {
      const damage = 1 - r.ans_exact_at_5;
      const cls = damage > 0.4 ? 'bad' : damage < 0.1 ? 'good' : '';
      return `<tr><td>${r.condition}</td><td>${(r.cer * 100).toFixed(2)}%</td>` +
        `<td>${pct(r.recall_at_5)}</td><td class="${cls}">${pct(r.ans_exact_at_5)}</td>` +
        `<td>${pct(r.ans_canon_at_5)}</td></tr>`;
    })
    .join('');
  el('table-wrap').innerHTML =
    `<table class="data"><thead><tr><th>condition</th><th>char error</th>` +
    `<th>recall@5</th><th>answer</th><th>canonicalized</th></tr></thead><tbody>${cells}</tbody></table>`;
}

function picker(node, items, current, onPick) {
  node.innerHTML = '';
  items.forEach(({ key, label }) => {
    const b = document.createElement('button');
    b.textContent = label;
    b.setAttribute('aria-pressed', String(key === current()));
    b.addEventListener('click', () => {
      onPick(key);
      [...node.children].forEach((c) => c.setAttribute('aria-pressed', String(c === b)));
      render();
      buildTable();
    });
    node.appendChild(b);
  });
}

async function main() {
  const res = await fetch('./data/noise_bench.json');
  if (!res.ok) {
    el('banner').textContent = `Could not load the measurements (HTTP ${res.status}).`;
    return;
  }
  const data = await res.json();
  state.rows = data.rows;

  const retrievers = [...new Set(state.rows.map((r) => r.retriever))].map((k) => ({ key: k, label: k }));
  picker(el('family'), FAMILIES, () => state.family, (k) => { state.family = k; state.step = 0; });
  picker(el('retriever'), retrievers, () => state.retriever, (k) => { state.retriever = k; });

  el('scrub').addEventListener('input', (e) => { state.step = Number(e.target.value); render(); });
  window.addEventListener('resize', render);

  el('cap-what').textContent =
    `${data.meta.n_questions} questions over ${data.meta.n_chunks} chunks`;
  el('cap-where').textContent =
    `${data.meta.peak_rss_mb_serving} MB serving, NumPy ${data.meta.numpy}, no PyTorch`;

  // Land on the case the whole repository is about rather than on the baseline.
  const points = series();
  state.step = Math.max(points.findIndex((p) => p.condition === 'ocr cer~5%'), 0);
  render();
  buildTable();
}

main();
