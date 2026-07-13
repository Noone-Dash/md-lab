/* MD Lab — compute monitor */
"use strict";
const $ = (s, r = document) => r.querySelector(s);

const BUDGET_FIELDS = [
  ["max_concurrent", "Max concurrent jobs", 1, 8],
  ["max_gpu_jobs", "Max GPU jobs at once", 1, 4],
  ["cores_per_job", "CPU cores per job", 1, 20],
  ["mem_per_job_gb", "Memory per job (GB)", 2, 120],
];
let BUDGET_BUILT = false;

function bar(pct, cls) {
  pct = Math.max(0, Math.min(100, pct || 0));
  return `<div class="bar"><div class="bar-fill ${cls || ''}" style="width:${pct}%"></div></div>`;
}

function renderTele(t) {
  $("#backend").textContent = "backend: " + t.backend;
  const g = t.gpu || {};
  const cards = [
    { k: "GPU", v: g.name || "—", rows: [
      ["utilisation", `${g.util ?? 0}%`, bar(g.util, "g")],
      ["temperature", `${g.temp ?? "?"} °C`, ""],
      ["power", `${g.power ?? "?"} W`, ""],
    ], note: "memory is unified (shared with system RAM)" },
    { k: "CPU", v: `${t.cores} cores`, rows: [
      ["utilisation", `${t.cpu_pct}%`, bar(t.cpu_pct, "c")],
      ["load (1m)", `${t.load1}`, ""],
    ] },
    { k: "Memory (unified)", v: `${t.ram_used_gb} / ${t.ram_total_gb} GB`, rows: [
      ["used", `${t.ram_pct}%`, bar(t.ram_pct, "m")],
    ], note: "CPU + GPU share this on GB10" },
    { k: "Queue", v: "", rows: [
      ["running", `${t.counts.running}`, ""],
      ["queued", `${t.counts.queued}`, ""],
      ["paused", `${t.counts.paused}`, ""],
    ] },
  ];
  $("#tele").innerHTML = cards.map(c => `
    <div class="tcard">
      <div class="tk">${c.k}</div>
      <div class="tv">${c.v}</div>
      ${c.rows.map(([a, b, bb]) => `<div class="trow"><span>${a}</span><b>${b}</b></div>${bb}`).join("")}
      ${c.note ? `<div class="tnote">${c.note}</div>` : ""}
    </div>`).join("");
}

function buildBudget(b) {
  if (BUDGET_BUILT) return;
  BUDGET_BUILT = true;
  $("#budget-grid").innerHTML = BUDGET_FIELDS.map(([k, label, mn, mx]) =>
    `<div class="pfield"><label>${label}</label>
       <input type="number" id="b_${k}" min="${mn}" max="${mx}" value="${b[k]}"></div>`).join("");
  $("#apply-budget").onclick = async () => {
    const body = {};
    BUDGET_FIELDS.forEach(([k]) => body[k] = +$("#b_" + k).value);
    await fetch("/api/budget", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body) });
    tick();
  };
}
function updateDerived(b, cores) {
  const pct = Math.round(b.cores_per_job * b.max_concurrent / cores * 100);
  $("#budget-derived").textContent = `≈ up to ${pct}% of the machine`;
}

const STATE_ORDER = { running: 0, paused: 1, queued: 2, done: 3, error: 4, killed: 5 };
function actions(j) {
  const btn = (a, label, cls) =>
    `<button class="jbtn ${cls || ''}" data-job="${j.id}" data-act="${a}">${label}</button>`;
  if (j.state === "running") return btn("pause", "⏸ pause") + btn("kill", "✕ kill", "danger");
  if (j.state === "paused") return btn("resume", "▶ resume") + btn("kill", "✕ kill", "danger");
  if (j.state === "queued") return btn("kill", "✕ cancel", "danger");
  return `<a class="jbtn" href="/track/${j.track}?run=${j.id}">open →</a>`;
}
function renderJobs(jobs) {
  if (!jobs.length) { $("#reg-table").innerHTML = `<p class="muted">No jobs yet.</p>`; return; }
  jobs.sort((a, b) => (STATE_ORDER[a.state] - STATE_ORDER[b.state]));
  $("#reg-table").innerHTML = `
    <div class="reg-row reg-hdr">
      <span>state</span><span>experiment</span><span>track</span><span>gpu</span>
      <span>cores·mem</span><span>controls</span></div>` +
    jobs.map(j => `
    <div class="reg-row">
      <span class="status ${j.state}"><span class="dot"></span>${j.state}</span>
      <span><a href="/track/${j.track}?run=${j.id}">${j.name}</a>
        <div class="ri-time">${j.id}</div>${j.error ? `<div class="jerr">${j.error}</div>` : ""}</span>
      <span>${j.track}</span>
      <span>${j.needs_gpu ? "🟢 GPU" : "CPU"}</span>
      <span>${j.state === "queued" ? "—" : `${j.cores}c · ${j.mem_gb}G`}</span>
      <span class="jctrls">${actions(j)}</span>
    </div>`).join("");
  document.querySelectorAll(".jbtn[data-act]").forEach(b => b.onclick = async () => {
    await fetch(`/api/job/${b.dataset.job}/${b.dataset.act}`, { method: "POST" });
    tick();
  });
}

async function tick() {
  const d = await fetch("/api/monitor").then(r => r.json()).catch(() => null);
  if (!d) return;
  renderTele(d.telemetry);
  buildBudget(d.telemetry.budget);
  updateDerived(d.telemetry.budget, d.telemetry.cores);
  renderJobs(d.jobs);
}

$("#clear").onclick = async () => { await fetch("/api/job/x/clear", { method: "POST" }); tick(); };
tick();
setInterval(tick, 1500);
