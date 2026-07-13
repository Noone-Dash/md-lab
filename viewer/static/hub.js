/* MD Lab — hub / landing page */
"use strict";
const $ = (s, r = document) => r.querySelector(s);

const MODE_LABEL = { live: "live", model: "simplified model", unavailable: "not installed" };

async function boot() {
  const meta = await fetch("/api/meta").then(r => r.json()).catch(() => ({}));
  if (meta.gmx_version) $("#gmxver").textContent = "GROMACS " + meta.gmx_version + " · CUDA (GB10)";

  const cat = await fetch("/api/catalog").then(r => r.json());
  const grid = $("#track-grid");
  grid.innerHTML = "";
  for (const t of cat) {
    const a = document.createElement("a");
    a.className = "track-card mode-" + t.mode;
    a.href = "/track/" + t.id;
    a.innerHTML = `
      <div class="tc-top">
        <span class="tc-icon">${t.icon}</span>
        <span class="badge ${t.mode}">${MODE_LABEL[t.mode]}</span>
      </div>
      <div class="tc-name">${t.name}</div>
      <div class="tc-engine">${t.engine}</div>
      <div class="tc-blurb">${t.blurb}</div>
      <div class="tc-foot">${t.count} experiment${t.count === 1 ? "" : "s"} →</div>`;
    grid.appendChild(a);
  }

  const runs = await fetch("/api/runs").then(r => r.json());
  const rr = $("#recent-runs");
  rr.innerHTML = runs.length ? "" : `<p class="muted">No runs yet — open a track and launch one.</p>`;
  for (const r of runs.slice(0, 12)) {
    const el = document.createElement("a");
    el.className = "recent-item";
    el.href = `/track/${r.track || "classical"}?run=${r.id}`;
    el.innerHTML = `<span class="status ${r.status}"><span class="dot"></span>${r.status}</span>
      <div class="ri-name">${r.recipe_name}</div>
      <div class="ri-time">${r.id}</div>`;
    rr.appendChild(el);
  }
}
boot();
