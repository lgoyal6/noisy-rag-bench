// The noise, actually applied.
//
// noise.py and retrieval.py are copied verbatim into docs/ and loaded through
// pyodide, so the functions damaging a reader's text are the ones that produced
// the conditions table below, and answer_present is the same check the
// benchmark scores with.
//
// The point of doing it live: character error rate is a bad proxy for damage,
// and the only convincing way to show that is to let someone watch a small
// error rate destroy an answer while a large one does nothing.
(() => {
  const el = (id) => document.getElementById(id);
  const FAMILIES = [
    ['ocr', 'OCR'],
    ['whitespace', 'Whitespace'],
    ['hyphenation', 'Hyphens'],
    ['header', 'Running headers'],
    ['redaction', 'Redaction'],
  ];
  const DEFAULT_TEXT =
`Section 4.2 Termination for Convenience. Either party may terminate this
agreement upon ninety (90) days written notice to the other party. Upon
termination, the Client shall pay all fees accrued through the effective
date of termination, and the Provider shall return all Client materials
within thirty (30) days.`;
  const DEFAULT_ANSWER = 'ninety (90) days';

  const st = { fam: 'ocr', rate: 20, api: null };

  function picker(node, items, current, onPick) {
    node.innerHTML = '';
    items.forEach(([key, label]) => {
      const b = document.createElement('button');
      b.textContent = label;
      b.setAttribute('aria-pressed', String(key === current()));
      b.addEventListener('click', () => {
        onPick(key);
        [...node.children].forEach((c) => c.setAttribute('aria-pressed', String(c === b)));
      });
      node.appendChild(b);
    });
  }

  function mark(id, hit) {
    const node = el(id);
    node.textContent = hit ? 'found' : 'gone';
    node.className = hit ? 'yes' : 'no';
  }

  function apply() {
    if (!st.api) return;
    const clean = el('n-clean').value;
    const answer = el('n-answer').value;
    let r;
    try {
      r = JSON.parse(st.api.run(JSON.stringify({
        text: clean, answer, family: st.fam, rate: st.rate / 100,
      })));
    } catch (e) {
      el('noise-banner').className = 'banner alarm';
      el('noise-banner').textContent = `The noise functions rejected that: ${e}`;
      return;
    }
    el('n-dirty').textContent = r.dirty;
    el('n-cer').textContent = `${(r.cer * 100).toFixed(1)}%`;
    mark('n-clean-hit', r.clean_hit);
    mark('n-dirty-hit', r.dirty_hit);
    mark('n-canon-hit', r.canon_hit);

    const b = el('noise-banner');
    if (!answer.trim()) {
      b.className = 'banner';
      b.textContent = 'Put the answer you would be looking for in the box below.';
    } else if (!r.clean_hit) {
      b.className = 'banner';
      b.textContent =
        'That string is not in the clean document either, so there is nothing for the noise ' +
        'to take away. Copy a phrase out of the passage above.';
    } else if (r.dirty_hit) {
      b.className = 'banner calm';
      b.textContent =
        `Still there, at ${(r.cer * 100).toFixed(1)}% character error. Push the amount up, or ` +
        `switch to OCR, and watch a smaller error rate do more damage than this one.`;
    } else if (r.canon_hit) {
      b.className = 'banner alarm';
      b.textContent =
        `Gone at ${(r.cer * 100).toFixed(1)}% character error, and canonicalizing puts it back. ` +
        `That is the eleven-line fix: same index, same model, the answer returns.`;
    } else {
      b.className = 'banner alarm';
      b.textContent =
        `Gone at ${(r.cer * 100).toFixed(1)}% character error, and canonicalizing does not ` +
        `recover it. This is the case a retrieval dashboard will still call a hit.`;
    }
  }

  async function boot() {
    try {
      const py = await loadPyodide();
      await py.loadPackage('numpy');   // retrieval.py imports it for the dense arm
      for (const f of ['noise.py', 'retrieval.py']) {
        py.FS.writeFile(f, await (await fetch(`./data/${f}`)).text());
      }
      const api = py.runPython(`
import json, random
import noise, retrieval

def _run(payload):
    q = json.loads(payload)
    rng = random.Random(7)          # fixed, so moving the slider is the only change
    fn = noise.FAMILIES[q["family"]]
    dirty = fn(q["text"], q["rate"], rng)
    ans = q["answer"]
    return json.dumps({
        "dirty": dirty,
        "cer": float(noise.char_error_rate(q["text"], dirty)),
        "clean_hit": bool(ans) and bool(retrieval.answer_present(ans, [q["text"]])),
        "dirty_hit": bool(ans) and bool(retrieval.answer_present(ans, [dirty])),
        "canon_hit": bool(ans) and bool(retrieval.answer_present(ans, [dirty], canonicalize=True)),
    })

{"run": _run}
`).toJs({ dict_converter: Object.fromEntries });
      st.api = api;
      el('noise-engine').textContent = 'noise.py running in your tab, via pyodide';
      apply();
    } catch (e) {
      el('noise-engine').textContent = 'the engine did not start';
      el('noise-banner').className = 'banner alarm';
      el('noise-banner').textContent = `Could not start the noise functions: ${e}`;
    }
  }

  picker(el('noise-fam'), FAMILIES, () => st.fam, (k) => { st.fam = k; apply(); });
  el('noise-rate').addEventListener('input', (e) => {
    st.rate = Number(e.target.value);
    el('noise-rate-val').textContent = `${st.rate}%`;
    apply();
  });
  el('n-clean').value = DEFAULT_TEXT;
  el('n-answer').value = DEFAULT_ANSWER;
  el('n-clean').addEventListener('input', apply);
  el('n-answer').addEventListener('input', apply);
  boot();
})();
