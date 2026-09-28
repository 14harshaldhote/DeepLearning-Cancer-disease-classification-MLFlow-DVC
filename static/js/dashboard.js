const $ = (sel) => document.querySelector(sel);
const pct = (v, digits = 1) => (v == null || Number.isNaN(v) ? "–" : `${(v * 100).toFixed(digits)}%`);
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const tooltip = $("#tooltip");

/* ---------- theme ---------- */
(function initTheme() {
  try {
    const saved = localStorage.getItem("theme");
    if (saved) document.documentElement.dataset.theme = saved;
  } catch (_) { /* storage unavailable */ }
  $("#theme-toggle").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (_) { /* ignore */ }
  });
})();

/* ---------- health ---------- */
async function loadHealth() {
  const el = $("#status");
  try {
    const res = await fetch("/health");
    const data = await res.json();
    el.textContent = data.model_loaded ? "● Model ready" : "Model not loaded";
    el.className = `pill ${data.model_loaded ? "pill-good" : "pill-bad"}`;
  } catch (_) {
    el.textContent = "API offline";
    el.className = "pill pill-bad";
  }
}

/* ---------- prediction ---------- */
let currentTruth = null;

async function predict(blob, filename, truth = null) {
  currentTruth = truth;
  $("#result-error").hidden = true;
  $("#result-empty").innerHTML = "<p>Analysing…</p>";
  $("#result-empty").hidden = false;
  $("#result").hidden = true;

  const form = new FormData();
  form.append("file", blob, filename);
  try {
    const res = await fetch("/api/predict?explain=true", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
    renderResult(data, URL.createObjectURL(blob));
    loadHistory();
  } catch (err) {
    $("#result-empty").hidden = true;
    $("#result-error").textContent = err.message;
    $("#result-error").hidden = false;
  }
}

function renderResult(data, originalUrl) {
  $("#result-empty").hidden = true;
  $("#result").hidden = false;
  $("#pred-label").textContent = data.label;

  const badge = $("#review-badge");
  if (data.needs_review) {
    badge.className = "badge review";
    badge.textContent = `⚠ Needs expert review (below ${pct(data.review_threshold, 0)})`;
  } else {
    badge.className = "badge confident";
    badge.textContent = `✓ Confident (${pct(data.confidence)})`;
  }

  const bars = Object.entries(data.probabilities)
    .sort((a, b) => b[1] - a[1])
    .map(([label, p]) => `
      <div class="prob-row ${label === data.label ? "" : "dim"}">
        <span>${escapeHtml(label)}</span>
        <div class="prob-track"><div class="prob-fill" style="width:${(p * 100).toFixed(1)}%"></div></div>
        <span class="prob-value">${pct(p)}</span>
      </div>`).join("");
  $("#prob-bars").innerHTML = bars;

  $("#view-original").src = originalUrl;
  $("#view-heatmap").src = data.heatmap_png ? `data:image/png;base64,${data.heatmap_png}` : originalUrl;
  setView("heatmap");

  $("#meta-latency").textContent = `Inference ${data.latency_ms} ms (including Grad-CAM)`;
  if (currentTruth) {
    const correct = currentTruth === data.class;
    $("#meta-truth").textContent = `Ground truth: ${currentTruth} · ${correct ? "✓ correct" : "✗ incorrect"}`;
  } else {
    $("#meta-truth").textContent = "";
  }
}

function setView(view) {
  document.querySelectorAll(".seg").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  $("#view-original").hidden = view !== "original";
  $("#view-heatmap").hidden = view !== "heatmap";
}
document.querySelectorAll(".seg").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));

function initUpload() {
  const zone = $("#dropzone");
  const input = $("#file-input");
  input.addEventListener("change", () => input.files[0] && predict(input.files[0], input.files[0].name));
  zone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  ["dragenter", "dragover"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.remove("drag"); }));
  zone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) predict(file, file.name);
  });
}

async function loadSamples() {
  const samples = await (await fetch("/api/samples")).json();
  $("#samples").innerHTML = samples.map((s, i) => `
    <button type="button" class="sample" data-i="${i}">
      <img src="${s.url}" alt="${escapeHtml(s.true_class)} sample" loading="lazy">
      <span>${escapeHtml(s.true_class === "normal" ? "Normal" : "Adenocarcinoma")} · #${s.name.split("_").pop()}</span>
    </button>`).join("");
  document.querySelectorAll(".sample").forEach((btn) => btn.addEventListener("click", async () => {
    const s = samples[btn.dataset.i];
    const blob = await (await fetch(s.url)).blob();
    predict(blob, `${s.name}.png`, s.true_class);
  }));
}

/* ---------- model info ---------- */
async function loadModelInfo() {
  const info = await (await fetch("/api/model-info")).json();
  const ev = info.evaluation;
  if (!ev) {
    $("#kpis").innerHTML = `<p class="muted">No evaluation report yet. Run <code>dvc repro</code> to train and evaluate.</p>`;
  } else {
    $("#eval-date").textContent = `${ev.n_samples} test images · evaluated ${ev.evaluated_at.slice(0, 10)}`;
    const kpis = [
      ["Accuracy", ev.accuracy, "share of test images classified correctly"],
      ["Sensitivity", ev.sensitivity, "cancer slices caught (recall)"],
      ["Specificity", ev.specificity, "normal slices correctly cleared"],
      ["ROC-AUC", ev.roc_auc, "ranking quality across thresholds"],
      ["Macro F1", ev.macro_f1, "balance of precision and recall"],
    ];
    $("#kpis").innerHTML = kpis.map(([l, v, h]) => `
      <div class="kpi"><div class="label">${l}</div><div class="value">${l === "ROC-AUC" ? (v ?? 0).toFixed(3) : pct(v)}</div><div class="hint">${h}</div></div>`).join("");
    renderConfusion(ev);
  }
  if (info.training_history) {
    const h = info.training_history;
    lineChart($("#chart-loss"), h, "loss", "val_loss", (v) => v.toFixed(2));
    lineChart($("#chart-acc"), h, "accuracy", "val_accuracy", (v) => `${Math.round(v * 100)}%`, [0, 1]);
    renderHistoryTable(h);
  }
  renderPipeline(info);
}

function renderConfusion(ev) {
  const names = ev.class_names.map((n) => (n === "normal" ? "Normal" : "Adenocarcinoma"));
  const cm = ev.confusion_matrix;
  const max = Math.max(...cm.flat(), 1);
  const colour = (v) => {
    const t = v / max;
    if (t === 0) return ["var(--surface-2)", "var(--ink-2)"];
    if (t < 0.34) return ["var(--seq-1)", "var(--ink)"];
    if (t < 0.67) return ["var(--seq-2)", "var(--ink)"];
    return ["var(--seq-3)", "#fff"];
  };
  let html = `<div></div>${names.map((n) => `<div class="hdr">${n}</div>`).join("")}`;
  cm.forEach((row, i) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    html += `<div class="rowhdr">Actual ${names[i]}</div>`;
    row.forEach((v, j) => {
      const [bg, fg] = colour(v);
      html += `<div class="cell" style="background:${bg};color:${fg}" data-tip="Actual ${names[i]} → predicted ${names[j]}: ${v} images (${pct(v / total)} of row)">${v}<small>${pct(v / total, 0)}</small></div>`;
    });
  });
  html += `<div></div><div class="axis-label">Predicted class</div>`;
  $("#confusion").innerHTML = html;
  $("#confusion").querySelectorAll(".cell").forEach((c) => {
    c.addEventListener("mousemove", (e) => showTip(e, `<div>${escapeHtml(c.dataset.tip)}</div>`));
    c.addEventListener("mouseleave", hideTip);
  });
}

/* ---------- line chart (two series, one axis) ---------- */
function lineChart(el, history, trainKey, valKey, fmt, fixedDomain) {
  const train = history[trainKey] || [];
  const val = history[valKey] || [];
  const n = Math.max(train.length, val.length);
  if (!n) return;
  const W = 360, H = 200, m = { t: 10, r: 44, b: 26, l: 40 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const all = [...train, ...val];
  let [lo, hi] = fixedDomain || [Math.min(...all), Math.max(...all)];
  if (!fixedDomain) { const pad = (hi - lo) * 0.08 || 0.1; lo = Math.max(0, lo - pad); hi += pad; }
  const x = (i) => m.l + (n === 1 ? iw / 2 : (i / (n - 1)) * iw);
  const y = (v) => m.t + ih - ((v - lo) / (hi - lo)) * ih;
  const path = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => lo + t * (hi - lo));
  const xticks = [...new Set([0, Math.floor((n - 1) / 2), n - 1])];
  const best = history.best_epoch ? history.best_epoch - 1 : null;

  const labelY = (arr, other) => {
    let yy = y(arr[arr.length - 1]);
    const yo = y(other[other.length - 1]);
    if (Math.abs(yy - yo) < 12) yy += yy < yo ? -6 : 6;
    return yy + 4;
  };

  el.innerHTML = `
    <div class="legend"><span><i style="background:var(--series-1)"></i>Train</span><span><i style="background:var(--series-2)"></i>Validation</span></div>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${trainKey} and ${valKey} per epoch">
      ${ticks.map((t) => `<line class="gridline" x1="${m.l}" x2="${W - m.r}" y1="${y(t)}" y2="${y(t)}"/><text x="${m.l - 6}" y="${y(t) + 4}" text-anchor="end">${fmt(t)}</text>`).join("")}
      <line class="axis" x1="${m.l}" x2="${W - m.r}" y1="${m.t + ih}" y2="${m.t + ih}"/>
      ${xticks.map((i) => `<text x="${x(i)}" y="${H - 8}" text-anchor="middle">${i + 1}</text>`).join("")}
      ${best != null && best < n ? `<line class="best" x1="${x(best)}" x2="${x(best)}" y1="${m.t}" y2="${m.t + ih}"/><text x="${x(best) + 4}" y="${m.t + 10}">best epoch ${best + 1}</text>` : ""}
      <path d="${path(train)}" fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round"/>
      <path d="${path(val)}" fill="none" stroke="var(--series-2)" stroke-width="2" stroke-linejoin="round"/>
      <text class="series-label" x="${x(n - 1) + 6}" y="${labelY(train, val)}">Train</text>
      <text class="series-label" x="${x(n - 1) + 6}" y="${labelY(val, train)}">Val</text>
      <line class="crosshair" y1="${m.t}" y2="${m.t + ih}" visibility="hidden"/>
      <circle class="dot-1" r="4" fill="var(--series-1)" stroke="var(--surface)" stroke-width="2" visibility="hidden"/>
      <circle class="dot-2" r="4" fill="var(--series-2)" stroke="var(--surface)" stroke-width="2" visibility="hidden"/>
      <rect x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent"/>
    </svg>`;

  const svg = el.querySelector("svg");
  const hit = svg.querySelector("rect");
  const cross = svg.querySelector(".crosshair");
  const d1 = svg.querySelector(".dot-1"), d2 = svg.querySelector(".dot-2");
  hit.addEventListener("mousemove", (e) => {
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - m.l) / iw) * (n - 1))));
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
    if (train[i] != null) { d1.setAttribute("cx", x(i)); d1.setAttribute("cy", y(train[i])); d1.setAttribute("visibility", "visible"); }
    if (val[i] != null) { d2.setAttribute("cx", x(i)); d2.setAttribute("cy", y(val[i])); d2.setAttribute("visibility", "visible"); }
    showTip(e, `<div><strong>Epoch ${i + 1}</strong></div>
      <div class="row"><span><i style="background:var(--series-1)"></i>Train</span><span>${train[i] != null ? fmt(train[i]) : "–"}</span></div>
      <div class="row"><span><i style="background:var(--series-2)"></i>Validation</span><span>${val[i] != null ? fmt(val[i]) : "–"}</span></div>`);
  });
  hit.addEventListener("mouseleave", () => {
    [cross, d1, d2].forEach((n2) => n2.setAttribute("visibility", "hidden"));
    hideTip();
  });
}

function renderHistoryTable(h) {
  const rows = h.loss.map((_, i) => `<tr><td class="num">${i + 1}</td><td class="num">${h.loss[i].toFixed(4)}</td><td class="num">${h.val_loss[i].toFixed(4)}</td><td class="num">${pct(h.accuracy[i])}</td><td class="num">${pct(h.val_accuracy[i])}</td></tr>`).join("");
  $("#history-table").innerHTML = `<div class="table-scroll"><table><thead><tr><th class="num">Epoch</th><th class="num">Loss</th><th class="num">Val loss</th><th class="num">Accuracy</th><th class="num">Val accuracy</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function renderPipeline(info) {
  const split = info.data_split?.classes || {};
  const sum = (k) => Object.values(split).reduce((a, c) => a + (c[k] || 0), 0);
  const h = info.training_history;
  const ev = info.evaluation;
  const steps = [
    ["Data ingestion", `${sum("raw_files") || "–"} CT slices, 2 classes, from the bundled zip or Google Drive`],
    ["Data preparation", `${sum("duplicates_removed")} duplicate files removed; split by scan id into ${sum("train")} / ${sum("val")} / ${sum("test")} train / val / test`],
    ["Base model", `VGG16 with ImageNet weights, conv layers frozen, pooled softmax head (Adam)`],
    ["Training", h ? `${h.loss.length} epochs with early stopping (best: ${h.best_epoch}), augmentation, class weights` : "Early stopping, augmentation, class weights"],
    ["Evaluation", ev ? `Test accuracy ${pct(ev.accuracy)}; logged to MLflow; quality gate ${ev.promotion?.passed ? "passed" : "not passed"}` : "Held-out test metrics logged to MLflow"],
    ["Serving", "FastAPI + Grad-CAM, Docker image, CI with GitHub Actions"],
  ];
  $("#pipeline").innerHTML = steps.map(([t, d]) => `<li><strong>${t}</strong><span>${escapeHtml(d)}</span></li>`).join("");
}

/* ---------- history ---------- */
async function loadHistory() {
  const rows = await (await fetch("/api/history")).json();
  if (!rows.length) {
    $("#history").innerHTML = `<p class="muted small">No predictions yet in this session.</p>`;
    return;
  }
  $("#history").innerHTML = `<div class="table-scroll"><table>
    <thead><tr><th>Time (UTC)</th><th>Image</th><th>Prediction</th><th class="num">Confidence</th><th>Review</th><th class="num">Latency</th></tr></thead>
    <tbody>${rows.map((r) => `<tr>
      <td>${r.time.slice(11, 19)}</td><td>${escapeHtml(r.source)}</td><td>${escapeHtml(r.label)}</td>
      <td class="num">${pct(r.confidence)}</td><td>${r.needs_review ? "⚠ Needs review" : "✓ Confident"}</td>
      <td class="num">${r.latency_ms} ms</td></tr>`).join("")}</tbody></table></div>`;
}

/* ---------- tooltip ---------- */
function showTip(e, html) {
  tooltip.innerHTML = html;
  tooltip.hidden = false;
  const pad = 14;
  const { innerWidth: vw } = window;
  const w = tooltip.offsetWidth;
  tooltip.style.left = `${Math.min(e.clientX + pad, vw - w - 8)}px`;
  tooltip.style.top = `${e.clientY + pad}px`;
}
function hideTip() { tooltip.hidden = true; }

/* ---------- boot ---------- */
initUpload();
loadHealth();
loadSamples();
loadModelInfo();
loadHistory();
