/* Plan Builder — write a simulation as JSON, see it checked, costed, and run. */
"use strict";
const $ = s => document.querySelector(s);

const EXAMPLES = {
  "Water box (fast)": {
    name: "SPC/E water, 310 K, NPT",
    system: { kind: "solvent", forcefield: "amber99sb-ildn", water_model: "spce",
              box_size_nm: 3.0, salt_conc_M: 0.15, neutralize: true },
    stages: [
      { name: "minimize", type: "minimize", max_steps: 5000 },
      { name: "production", type: "dynamics", sim_time_ns: 0.05,
        params: { ensemble: "NPT", temperature: 310 } }
    ],
    analyses: ["rdf_ow", "msd_ow"]
  },
  "Lysozyme in salt water": {
    name: "Lysozyme 310 K, 0.15 M NaCl",
    system: { kind: "protein", structure_source: "rcsb", pdb_id: "1AKI",
              forcefield: "amber99sb-ildn", water_model: "tip3p",
              box_shape: "dodecahedron", box_padding_nm: 1.2,
              salt_conc_M: 0.15, neutralize: true },
    stages: [
      { name: "minimize", type: "minimize", max_steps: 5000 },
      { name: "nvt", type: "dynamics", sim_time_ns: 0.02, posres_fc_kj: 1000,
        params: { ensemble: "NVT", temperature: 310 } },
      { name: "npt", type: "dynamics", sim_time_ns: 0.02, posres_fc_kj: 1000,
        params: { ensemble: "NPT", temperature: 310 } },
      { name: "production", type: "dynamics", sim_time_ns: 0.05,
        params: { ensemble: "NPT", temperature: 310 } }
    ],
    analyses: ["rmsd", "gyrate", "rmsf"]
  },
  "Deliberately broken (watch it get caught)": {
    name: "This plan is wrong on purpose",
    system: { kind: "solvent", box_size_nm: 1.8, neutralize: false },
    stages: [
      { name: "production", type: "dynamics", sim_time_ns: 0.05,
        params: { ensemble: "NPT", pcoupltype: "semiisotropic",
                  constraints: "none", dt: 0.002,
                  rvdw: 1.2, rcoulomb: 1.2, temperature: 300 } }
    ]
  }
};

function plan() {
  try { return JSON.parse($("#editor").value); }
  catch (e) { throw new Error("That isn't valid JSON: " + e.message); }
}
function setPlan(p) {
  $("#editor").value = JSON.stringify(p, null, 2);
  $("#plan-title").textContent = p.name || "untitled plan";
  validate(); estimateCost();
}

/* ---------- physics check ---------- */
async function validate() {
  const box = $("#validation");
  let p;
  try { p = plan(); } catch (e) { box.innerHTML = `<span class="v-err">${e.message}</span>`; return null; }
  const r = await fetch("/api/plan/validate", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plan: p })
  }).then(x => x.json());

  if (r.ok && !r.findings.length) {
    box.innerHTML = `<div class="v-ok">✓ Physics looks sound — nothing to flag.
      <span class="muted">(checked against ${r.n_rules} rules)</span></div>`;
  } else {
    box.innerHTML = r.findings.map(f => `
      <div class="v-item v-${f.severity}">
        <div class="v-head">${f.severity === "error" ? "✕" : "⚠"} ${f.message}</div>
        <div class="v-fix"><b>Fix:</b> ${f.fix}</div>
        <div class="v-rule">${f.rule}</div>
      </div>`).join("") +
      `<div class="muted" style="margin-top:8px">${r.errors} error(s), ${r.warnings} warning(s) — of ${r.n_rules} rules checked.</div>`;
  }
  $("#btn-run").disabled = !r.ok;
  $("#btn-run").textContent = r.ok ? "▶ Run it" : "✕ fix the errors first";
  return r;
}

/* ---------- cost ---------- */
async function estimateCost() {
  let p; try { p = plan(); } catch { return; }
  const c = await fetch("/api/plan/estimate", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plan: p })
  }).then(x => x.json());
  if (c.error) { $("#cost").textContent = "—"; return; }
  $("#cost").innerHTML = `
    <div class="cost-big">${c.total_human}</div>
    <div class="muted" style="font-size:12px;margin-bottom:8px">
      ~${c.n_atoms_estimated.toLocaleString()} atoms · ~${c.peak_memory_gb_estimated} GB ·
      ${c.throughput_source}</div>` +
    c.per_stage.map(s => `<div class="lg-row"><span>${s.stage}</span><b>${s.seconds}s</b></div>`).join("");
}

/* ---------- mdp preview ---------- */
async function showMdp() {
  let p; try { p = plan(); } catch (e) { return; }
  const r = await fetch("/api/plan/mdp", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plan: p })
  }).then(x => x.json());
  $("#mdp-panel").classList.remove("hidden");
  $("#mdp").textContent = (r.stages || [])
    .map(s => `### ${s.name} (${s.type})\n${s.mdp}`).join("\n");
}

/* ---------- run ---------- */
async function run() {
  let p; try { p = plan(); } catch (e) { return; }
  const btn = $("#btn-run"); btn.disabled = true; btn.textContent = "launching…";
  const r = await fetch("/api/plan/run", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plan: p })
  }).then(x => x.json());
  if (r.submitted) location.href = `/track/plan?run=${r.run_id}`;
  else { btn.disabled = false; btn.textContent = "▶ Run it"; validate(); }
}

/* ---------- ontology search ---------- */
let searchTimer = null;
$("#onto-search").addEventListener("input", e => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(async () => {
    const q = e.target.value.trim();
    if (!q) { $("#onto-results").innerHTML = ""; return; }
    const r = await fetch("/api/ontology?search=" + encodeURIComponent(q)).then(x => x.json());
    $("#onto-results").innerHTML =
      `<div class="muted" style="font-size:11px;margin:6px 0">${r.count} match(es)</div>` +
      r.parameters.slice(0, 8).map(p => `
        <div class="onto-item">
          <div class="onto-key">${p.key} <span class="onto-unit">${p.unit || ""}</span></div>
          <div class="onto-mean">${p.meaning}</div>
          ${p.agent_guidance ? `<div class="onto-guide">→ ${p.agent_guidance}</div>` : ""}
        </div>`).join("");
  }, 250);
});

/* ---------- boot ---------- */
$("#examples").innerHTML = Object.keys(EXAMPLES).map((k, i) =>
  `<button class="ex-btn" data-k="${k}">${k}</button>`).join("");
document.querySelectorAll(".ex-btn").forEach(b =>
  b.onclick = () => {
    document.querySelectorAll(".ex-btn").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    setPlan(EXAMPLES[b.dataset.k]);
  });
$("#btn-validate").onclick = validate;
$("#btn-mdp").onclick = showMdp;
$("#btn-run").onclick = run;
$("#editor").addEventListener("input", () => {
  clearTimeout(window._t);
  window._t = setTimeout(() => { validate(); estimateCost(); }, 500);
});

document.querySelector(".ex-btn").classList.add("active");
setPlan(EXAMPLES["Lysozyme in salt water"]);
