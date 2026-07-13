/* MD Lab — per-track page controller */
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const TRACK = window.TRACK_ID;
let EXPERIMENTS = [];
let SELECTED = null;
let CURRENT = null;
let POLL = null;

const V = { category: "", loaded: null };

async function boot() {
  const t = await fetch("/api/track/" + TRACK).then(r => r.json());
  $("#track-engine").textContent = t.engine + "  ·  " +
    ({ live: "live engine", model: "simplified model", unavailable: "placeholder" }[t.mode] || "");
  EXPERIMENTS = t.experiments;
  renderCards();
  wireTabs();
  V3D.init($("#viewer-stage"));
  refreshRuns();

  const q = new URLSearchParams(location.search).get("run");
  if (q) openRun(q, null);
}

/* ---------- tabs ---------- */
function wireTabs() {
  $$(".tab[data-tab]").forEach(t => t.onclick = () => {
    $$(".tab").forEach(x => x.classList.remove("active"));
    t.classList.add("active");
    const which = t.dataset.tab;
    $("#panel-new").classList.toggle("hidden", which !== "new");
    $("#panel-runs").classList.toggle("hidden", which !== "runs");
    if (which === "runs") refreshRuns();
  });
}

/* ---------- experiment cards ---------- */
const MODE_TXT = { live: "live", model: "model", unavailable: "not installed" };
function renderCards() {
  const box = $("#recipe-cards");
  box.innerHTML = "";
  // group experiments by classification
  const groups = {};
  for (const r of EXPERIMENTS) (groups[r.classification || "other"] ||= []).push(r);
  const multi = Object.keys(groups).length > 1;
  for (const [cls, items] of Object.entries(groups)) {
    if (multi) {
      const h = document.createElement("div");
      h.className = "class-head";
      h.textContent = cls;
      box.appendChild(h);
    }
    for (const r of items) {
      const c = document.createElement("div");
      c.className = "rcard mode-" + r.mode;
      c.innerHTML = `<span class="badge ${r.mode}">${MODE_TXT[r.mode]}</span>
        <div class="rname">${r.name}</div>
        <div class="rdesc">${r.description}</div>
        <div class="rest">${r.engine} · ≈ ${r.est}</div>`;
      c.onclick = () => selectExperiment(r, c);
      box.appendChild(c);
    }
  }
}

function selectExperiment(r, card) {
  SELECTED = r;
  $$(".rcard").forEach(x => x.classList.remove("active"));
  card.classList.add("active");
  const form = $("#param-form");
  form.classList.remove("hidden");
  $("#form-title").textContent = r.name;
  $("#form-desc").textContent = r.description;
  const fields = $("#param-fields");
  const note = $("#unavailable-note");
  const launch = $(".launch");

  if (r.mode === "unavailable") {
    fields.innerHTML = "";
    note.classList.remove("hidden");
    note.innerHTML = `⚠ ${r.description}<br><br>Ask to install it and this becomes live.`;
    launch.disabled = true;
    launch.textContent = "unavailable";
    return;
  }
  note.classList.add("hidden");
  launch.disabled = false;
  launch.textContent = "▶ Launch";
  fields.innerHTML = "";
  for (const p of r.params) fields.appendChild(paramField(p));
}

function paramField(p) {
  const wrap = document.createElement("div");
  wrap.className = "pfield";
  const id = "p_" + p.name;
  let control;
  if (p.type === "choice") {
    control = `<select id="${id}" data-name="${p.name}">` +
      p.options.map(o => `<option value="${o}">${o}</option>`).join("") + `</select>`;
  } else if (p.type === "bool") {
    control = `<label class="switch"><input type="checkbox" id="${id}" data-name="${p.name}" data-type="bool" ${p.default ? "checked" : ""}> enabled</label>`;
  } else if ((p.type === "int" || p.type === "float") && p.min != null) {
    const step = p.step || (p.type === "int" ? 1 : 0.1);
    control = `<div class="rangewrap">
        <input type="range" id="${id}" data-name="${p.name}" data-type="${p.type}"
               min="${p.min}" max="${p.max}" step="${step}" value="${p.default}">
        <span class="rval" id="${id}_v">${p.default}</span></div>`;
  } else {
    control = `<input type="number" id="${id}" data-name="${p.name}" data-type="${p.type}" value="${p.default}">`;
  }
  wrap.innerHTML = `<label>${p.label}</label>${control}` +
    (p.help ? `<div class="phelp">${p.help}</div>` : "");
  setTimeout(() => {
    const el = $("#" + id);
    if (el && el.type === "range") el.oninput = () => { $("#" + id + "_v").textContent = el.value; };
  }, 0);
  return wrap;
}

$("#param-form").addEventListener("submit", async e => {
  e.preventDefault();
  if (!SELECTED || SELECTED.mode === "unavailable") return;
  const params = {};
  $$("#param-fields [data-name]").forEach(el => {
    params[el.dataset.name] = el.type === "checkbox" ? (el.checked ? "true" : "false") : el.value;
  });
  const btn = $(".launch"); btn.disabled = true; btn.textContent = "launching…";
  try {
    const res = await fetch("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key: SELECTED.key, params }),
    }).then(r => r.json());
    openRun(res.id, SELECTED.category);
  } finally {
    btn.disabled = false; btn.textContent = "▶ Launch";
  }
});

/* ---------- runs ---------- */
async function refreshRuns() {
  const runs = (await fetch("/api/runs").then(r => r.json()))
    .filter(r => (r.track || "classical") === TRACK);
  const box = $("#runs-list");
  box.innerHTML = runs.length ? "" : `<p class="muted">No runs in this track yet.</p>`;
  for (const r of runs) {
    const el = document.createElement("div");
    el.className = "run-item" + (r.id === CURRENT ? " active" : "");
    el.innerHTML = `<div><div class="ri-name">${r.recipe_name}</div>
      <div class="ri-time">${r.id}</div></div>
      <span class="status ${r.status}"><span class="dot"></span>${r.status}</span>`;
    el.onclick = () => openRun(r.id, r.category);
    box.appendChild(el);
  }
}

/* ---------- open + poll ---------- */
function openRun(id, category) {
  CURRENT = id;
  V.category = category || "";
  V.loaded = null;
  PLOTTED_STATUS = null;
  $("#empty-state").classList.add("hidden");
  $("#run-view").classList.remove("hidden");
  $("#plots").innerHTML = "";
  $("#summary").classList.add("hidden");
  if (POLL) clearInterval(POLL);
  pollRun();
  POLL = setInterval(pollRun, 1200);
}

async function pollRun() {
  if (!CURRENT) return;
  const m = await fetch("/api/run/" + CURRENT).then(r => r.json()).catch(() => null);
  if (!m) return;
  V.category = m.category || V.category;
  renderRun(m);
  fetch("/api/run/" + CURRENT + "/log").then(r => r.text()).then(t => {
    const log = $("#log"); log.textContent = t; log.scrollTop = log.scrollHeight;
  });
  if (m.status === "done" || m.status === "error") {
    clearInterval(POLL); POLL = null;
    refreshRuns();
  }
}

function renderRun(m) {
  $("#run-title").textContent = m.recipe_name;
  $("#run-engine").textContent = m.engine || "";
  $("#run-status").className = "status " + m.status;
  $("#run-status").innerHTML = `<span class="dot"></span>${m.status}`;
  $("#run-params").innerHTML = Object.entries(m.params)
    .map(([k, v]) => `<span class="chip">${k} <b>${v}</b></span>`).join("");
  $("#steps").innerHTML = (m.steps || []).map(s =>
    `<span class="step ${s.status}">${s.name}` +
    (s.seconds != null ? `<span class="secs">${s.seconds}s</span>` : "") + `</span>`).join("");

  // QM/MM (or any) key-value summary
  if (m.summary && m.summary.length) {
    const el = $("#summary"); el.classList.remove("hidden");
    el.innerHTML = m.summary.map(([k, v]) =>
      `<div class="sum-item"><span class="sk">${k}</span><span class="sv">${v}</span></div>`).join("");
  }

  if (m.outputs && m.outputs.trajectory_pdb && V.loaded !== CURRENT) {
    V.loaded = CURRENT;
    loadTrajectory(CURRENT, m);
  }
  renderPlots(m);
}

/* ---------- viewer: delegated to V3D (3Dmol + representations/legend/measure) ---- */
function viewerError(msg) {
  $("#viewer").innerHTML =
    `<div class="viewer-error">⚠ ${msg}<br><span>The run data is fine — this is a rendering problem.</span></div>`;
}

function captionFor(key) {
  const e = EXPERIMENTS.find(x => x.key === key);
  return (e && e.what_you_see) || "";
}

function showLoading(msg, pct, sub) {
  const el = $("#vloading");
  el.classList.remove("hidden");
  el.innerHTML = `<div>${msg}</div>
    <div class="vl-bar"><div class="vl-fill" style="width:${pct || 0}%"></div></div>
    <div class="vl-sub">${sub || ""}</div>`;
}
function hideLoading() { $("#vloading").classList.add("hidden"); }

/* Trajectories are megabytes of text. Stream them and SHOW the progress —
   a silent empty canvas for 20 s is indistinguishable from a broken viewer. */
async function loadTrajectory(runId, m) {
  showLoading("Loading trajectory…", 0, "contacting server");
  try {
    const r = await fetch("/api/run/" + runId + "/traj");
    if (!r.ok) throw new Error("no trajectory (HTTP " + r.status + ")");

    const total = +(r.headers.get("X-Uncompressed-Length") ||
                    r.headers.get("Content-Length") || 0);
    let data;
    if (r.body && total) {
      const reader = r.body.getReader();
      const chunks = [];
      let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        got += value.length;
        const pct = Math.min(99, Math.round(got / total * 100));
        showLoading("Loading trajectory…", pct,
                    `${(got / 1048576).toFixed(1)} MB of ~${(total / 1048576).toFixed(1)} MB`);
      }
      data = new TextDecoder().decode(
        chunks.reduce((acc, c) => { acc.set(c, acc._o || 0); acc._o = (acc._o || 0) + c.length; return acc; },
          Object.assign(new Uint8Array(chunks.reduce((n, c) => n + c.length, 0)), { _o: 0 })));
    } else {
      data = await r.text();
    }

    showLoading("Building the 3D scene…", 100, `${(data.length / 1048576).toFixed(1)} MB parsed`);
    await new Promise(res => setTimeout(res, 30));   // let the paint happen
    V3D.load(data, { category: m.category || "", caption: captionFor(m.recipe),
                     ligand: !!m.ligand });
    hideLoading();
  } catch (e) {
    hideLoading();
    viewerError(e.message || "could not render this trajectory");
  }
}

/* ---------- plots ---------- */
function findSeries(energy, name) {
  const i = energy.legends.findIndex(l => l.replace(/[\s.]/g, "").toLowerCase()
    === name.replace(/[\s.]/g, "").toLowerCase());
  return i >= 0 ? energy.series[i] : null;
}
function plotCard(title, help, spec) {
  const card = document.createElement("div");
  card.className = "plot-card";
  card.innerHTML = `<h4>${title}</h4>` + (help ? `<p class="phelp">${help}</p>` : "") + `<canvas></canvas>`;
  const legend = document.createElement("div");
  legend.className = "plot-legend";
  if (spec.legends && spec.legends.length > 1) {
    legend.innerHTML = spec.legends.map((l, i) =>
      `<span><i style="background:${window.PLOT_PALETTE[i % 7]}"></i>${l}</span>`).join("");
  }
  card.appendChild(legend);
  $("#plots").appendChild(card);
  requestAnimationFrame(() => drawLinePlot(card.querySelector("canvas"), spec));
}

let PLOTTED_STATUS = null;
function renderPlots(m) {
  const stamp = m.status + "|" + (m.energy ? m.energy.x.length : 0) + "|" + m.analyses.length;
  if (stamp === PLOTTED_STATUS) return;
  PLOTTED_STATUS = stamp;
  $("#plots").innerHTML = "";

  const e = m.energy;
  if (e) {
    const xlab = e.xaxis || "step";
    const named = { Potential: null, Kinetic: null, Total: null };
    const pot = findSeries(e, "Potential"), kin = findSeries(e, "Kinetic En.") || findSeries(e, "Kinetic"),
          tot = findSeries(e, "Total Energy") || findSeries(e, "Total");
    const es = [], el = [];
    if (pot) { es.push(pot); el.push("Potential"); }
    if (kin) { es.push(kin); el.push("Kinetic"); }
    if (tot) { es.push(tot); el.push("Total"); }
    if (es.length) plotCard("Energy", e.yaxis || "kJ/mol",
      { x: e.x, series: es, legends: el, xlabel: xlab, ylabel: e.yaxis || "kJ/mol" });

    const temp = findSeries(e, "Temperature");
    if (temp && !es.includes(temp)) plotCard("Temperature", "thermostat holding target",
      { x: e.x, series: [temp], xlabel: xlab, ylabel: "K" });
    const pres = findSeries(e, "Pressure");
    if (pres) plotCard("Pressure", "fluctuates strongly — normal",
      { x: e.x, series: [pres], xlabel: xlab, ylabel: "bar" });
    const dens = findSeries(e, "Density");
    if (dens) plotCard("Density", "equilibrating", { x: e.x, series: [dens], xlabel: xlab, ylabel: "kg/m³" });
  }

  for (const a of (m.analyses || [])) {
    const d = a.data;
    plotCard(a.label, a.help,
      { x: d.x, series: d.series, legends: d.legends, xlabel: d.xaxis, ylabel: d.yaxis });
  }
  if (!e && !(m.analyses || []).length && m.status !== "done" && m.status !== "error") {
    $("#plots").innerHTML = `<p class="muted">Plots appear once the run produces data…</p>`;
  }
}

boot();
