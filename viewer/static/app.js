/* GROMACS Lab — front-end controller */
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

let RECIPES = [];
let SELECTED = null;              // selected recipe meta
let CURRENT = null;               // current run id
let POLL = null;

const V = { viewer: null, nframes: 0, frame: 0, playing: false, timer: null,
            speed: 120, category: "", loaded: null };

/* ---------------- boot ---------------- */
async function boot() {
  const meta = await fetch("/api/meta").then(r => r.json()).catch(() => ({}));
  if (meta.gmx_version) $("#gmxver").textContent = "GROMACS " + meta.gmx_version;

  RECIPES = await fetch("/api/recipes").then(r => r.json());
  renderRecipeCards();
  wireTabs();
  wireViewerControls();
  refreshRuns();
}

/* ---------------- tabs ---------------- */
function wireTabs() {
  $$(".tab").forEach(t => t.onclick = () => {
    $$(".tab").forEach(x => x.classList.remove("active"));
    t.classList.add("active");
    const which = t.dataset.tab;
    $("#panel-new").classList.toggle("hidden", which !== "new");
    $("#panel-runs").classList.toggle("hidden", which !== "runs");
    if (which === "runs") refreshRuns();
  });
}

/* ---------------- recipe cards + form ---------------- */
function renderRecipeCards() {
  const box = $("#recipe-cards");
  box.innerHTML = "";
  for (const r of RECIPES) {
    const c = document.createElement("div");
    c.className = "rcard";
    c.innerHTML = `<span class="rcat">${r.category}</span>
      <div class="rname">${r.name}</div>
      <div class="rdesc">${r.description}</div>
      <div class="rest">runtime ≈ ${r.est}</div>`;
    c.onclick = () => selectRecipe(r, c);
    box.appendChild(c);
  }
}

function selectRecipe(r, card) {
  SELECTED = r;
  $$(".rcard").forEach(x => x.classList.remove("active"));
  card.classList.add("active");
  const form = $("#param-form");
  form.classList.remove("hidden");
  $("#form-title").textContent = r.name;
  $("#form-desc").textContent = r.description;
  const fields = $("#param-fields");
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
  // live value readout for sliders
  setTimeout(() => {
    const el = $("#" + id);
    if (el && el.type === "range") el.oninput = () => { $("#" + id + "_v").textContent = el.value; };
  }, 0);
  return wrap;
}

$("#param-form").addEventListener("submit", async e => {
  e.preventDefault();
  const params = {};
  $$("#param-fields [data-name]").forEach(el => { params[el.dataset.name] = el.value; });
  const btn = $(".launch"); btn.disabled = true; btn.textContent = "launching…";
  try {
    const res = await fetch("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ recipe: SELECTED.key, params }),
    }).then(r => r.json());
    openRun(res.id, SELECTED.category);
  } finally {
    btn.disabled = false; btn.textContent = "▶ Launch simulation";
  }
});

/* ---------------- runs list ---------------- */
async function refreshRuns() {
  const runs = await fetch("/api/runs").then(r => r.json());
  const box = $("#runs-list");
  box.innerHTML = runs.length ? "" : `<p class="muted">No runs yet.</p>`;
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

/* ---------------- open + poll a run ---------------- */
function openRun(id, category) {
  CURRENT = id;
  V.category = category || "";
  V.loaded = null;
  $("#empty-state").classList.add("hidden");
  $("#run-view").classList.remove("hidden");
  $("#plots").innerHTML = "";
  if (POLL) clearInterval(POLL);
  pollRun();
  POLL = setInterval(pollRun, 1200);
}

async function pollRun() {
  if (!CURRENT) return;
  const m = await fetch("/api/run/" + CURRENT).then(r => r.json()).catch(() => null);
  if (!m) return;
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
  $("#run-status").className = "status " + m.status;
  $("#run-status").innerHTML = `<span class="dot"></span>${m.status}`;

  $("#run-params").innerHTML = Object.entries(m.params)
    .map(([k, v]) => `<span class="chip">${k} <b>${v}</b></span>`).join("");

  $("#steps").innerHTML = (m.steps || []).map(s =>
    `<span class="step ${s.status}">${s.name}` +
    (s.seconds != null ? `<span class="secs">${s.seconds}s</span>` : "") + `</span>`).join("");

  // trajectory: load once available
  if (m.outputs && m.outputs.trajectory_pdb && V.loaded !== CURRENT) {
    V.loaded = CURRENT;
    loadTrajectory(CURRENT);
  }
  // plots
  renderPlots(m);
}

/* ---------------- 3Dmol viewer ---------------- */
function loadTrajectory(runId) {
  fetch("/api/run/" + runId + "/traj").then(r => { if (!r.ok) throw 0; return r.text(); })
    .then(data => {
      const el = $("#viewer");
      if (!V.viewer) V.viewer = $3Dmol.createViewer(el, { backgroundColor: "#0a0d12" });
      V.viewer.clear();
      V.viewer.addModelsAsFrames(data, "pdb");
      V.nframes = V.viewer.getNumFrames ? V.viewer.getNumFrames() : 1;
      applyStyle();
      if ($("#box").checked) { try { V.viewer.addUnitCell(V.viewer.getModel(0)); } catch (e) {} }
      V.viewer.zoomTo();
      V.viewer.render();
      V.frame = 0;
      const sl = $("#frame"); sl.max = Math.max(0, V.nframes - 1); sl.value = 0;
      updateFrameLabel();
      if (V.nframes > 1) play();
    }).catch(() => {});
}

function styleFor(kind) {
  if (kind === "auto") kind = (V.category === "Biomolecular") ? "cartoon" : "sphere";
  switch (kind) {
    case "sphere": return { sphere: { scale: 0.32 } };
    case "ballstick": return { stick: { radius: 0.13 }, sphere: { scale: 0.26 } };
    case "cartoon": return { cartoon: { color: "spectrum" }, stick: { radius: 0.1 } };
    case "line": return { line: {} };
    default: return { sphere: { scale: 0.32 } };
  }
}
function applyStyle() {
  if (!V.viewer) return;
  V.viewer.setStyle({}, styleFor($("#style").value));
  V.viewer.render();
}

function showFrame(i) {
  V.frame = i;
  if (V.viewer.setFrame) V.viewer.setFrame(i).then(() => V.viewer.render());
  else V.viewer.render();
  updateFrameLabel();
}
function updateFrameLabel() {
  $("#frame-label").textContent = `${V.frame + 1} / ${V.nframes}`;
  $("#frame").value = V.frame;
}
function play() {
  if (V.nframes < 2) return;
  V.playing = true; $("#play").textContent = "❚❚";
  clearInterval(V.timer);
  V.timer = setInterval(() => {
    const n = (V.frame + 1) % V.nframes;
    showFrame(n);
  }, V.speed);
}
function pause() { V.playing = false; $("#play").textContent = "▶"; clearInterval(V.timer); }

function wireViewerControls() {
  $("#play").onclick = () => V.playing ? pause() : play();
  $("#frame").oninput = e => { pause(); showFrame(+e.target.value); };
  $("#style").onchange = applyStyle;
  $("#speed").onchange = e => { V.speed = +e.target.value; if (V.playing) play(); };
  $("#spin").onchange = e => { if (V.viewer) { V.viewer.spin(e.target.checked ? "y" : false); } };
  $("#box").onchange = () => { if (V.loaded) { const id = V.loaded; V.loaded = null; loadTrajectory(id); V.loaded = id; } };
}

/* ---------------- plots ---------------- */
function findSeries(energy, name) {
  const i = energy.legends.findIndex(l => l.replace(/[\s.]/g, "").toLowerCase()
    === name.replace(/[\s.]/g, "").toLowerCase());
  return i >= 0 ? energy.series[i] : null;
}

function plotCard(title, help, spec) {
  const card = document.createElement("div");
  card.className = "plot-card";
  card.innerHTML = `<h4>${title}</h4>` + (help ? `<p class="phelp">${help}</p>` : "") +
    `<canvas></canvas>`;
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
  // only (re)draw when something changed materially
  const stamp = m.status + "|" + (m.energy ? m.energy.x.length : 0) + "|" + m.analyses.length;
  if (stamp === PLOTTED_STATUS) return;
  PLOTTED_STATUS = stamp;
  $("#plots").innerHTML = "";

  if (m.energy) {
    const e = m.energy, xlab = e.xaxis || "Time (ps)";
    const pot = findSeries(e, "Potential"), kin = findSeries(e, "Kinetic En."),
          tot = findSeries(e, "Total Energy");
    const enSeries = [], enLeg = [];
    if (pot) { enSeries.push(pot); enLeg.push("Potential"); }
    if (kin) { enSeries.push(kin); enLeg.push("Kinetic"); }
    if (tot) { enSeries.push(tot); enLeg.push("Total"); }
    if (enSeries.length)
      plotCard("Energy", "kJ/mol vs simulation time",
        { x: e.x, series: enSeries, legends: enLeg, xlabel: xlab, ylabel: "kJ/mol" });

    const temp = findSeries(e, "Temperature");
    if (temp) plotCard("Temperature", "thermostat holding the target T",
      { x: e.x, series: [temp], xlabel: xlab, ylabel: "K" });

    const pres = findSeries(e, "Pressure");
    if (pres) plotCard("Pressure", "fluctuates a lot — that's normal",
      { x: e.x, series: [pres], xlabel: xlab, ylabel: "bar" });

    const dens = findSeries(e, "Density");
    if (dens) plotCard("Density", "equilibrating toward the model's value",
      { x: e.x, series: [dens], xlabel: xlab, ylabel: "kg/m³" });
  }

  for (const a of (m.analyses || [])) {
    const d = a.data;
    plotCard(a.label, a.help,
      { x: d.x, series: d.series, legends: d.legends,
        xlabel: d.xaxis, ylabel: d.yaxis });
  }

  if (!m.energy && !(m.analyses || []).length && m.status !== "done") {
    $("#plots").innerHTML = `<p class="muted">Plots appear once the run produces data…</p>`;
  }
}

boot();
