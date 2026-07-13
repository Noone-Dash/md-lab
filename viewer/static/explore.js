/* Structure Explorer — fetch any real structure from RCSB/AlphaFold and render it.
   Streamed through the backend (no CORS pain, nothing kept in memory server-side). */
"use strict";
const $ = s => document.querySelector(s);

const EXAMPLES = [
  { db: "pdb", id: "1AKI", label: "Lysozyme", note: "classic enzyme" },
  { db: "pdb", id: "1UBQ", label: "Ubiquitin", note: "76 aa" },
  { db: "pdb", id: "1L2Y", label: "Trp-cage", note: "mini-protein" },
  { db: "pdb", id: "4HHB", label: "Haemoglobin", note: "4 chains + haem" },
  { db: "pdb", id: "6LU7", label: "SARS-CoV-2 protease", note: "+ inhibitor" },
  { db: "pdb", id: "1BNA", label: "B-DNA", note: "double helix" },
  { db: "alphafold", id: "P69905", label: "AlphaFold: haemoglobin α", note: "predicted" },
];

let viewer = null;

function styleSpec(kind) {
  switch (kind) {
    case "cartoon": return { cartoon: { color: "spectrum" } };
    case "ballstick": return { stick: { radius: 0.15 }, sphere: { scale: 0.25 } };
    case "sphere": return { sphere: { scale: 0.35 } };
    default: return { cartoon: { color: "spectrum" } };
  }
}

async function load(db, id) {
  const st = $("#status");
  st.textContent = `fetching ${id.toUpperCase()} …`;
  $("#info").innerHTML = "";
  try {
    const r = await fetch(`/api/structure/${db}/${encodeURIComponent(id)}`);
    if (!r.ok) throw new Error(`not found (HTTP ${r.status})`);
    const data = await r.text();

    if (!viewer) viewer = $3Dmol.createViewer($("#exp-viewer"), { backgroundColor: "#0a0d12" });
    viewer.clear();
    viewer.addModel(data, "pdb", { keepH: true });   // 3Dmol drops hydrogens otherwise

    const kind = $("#style").value;
    if (kind === "surface") {
      viewer.setStyle({}, { cartoon: { color: "spectrum" } });
      viewer.addSurface($3Dmol.SurfaceType.VDW, { opacity: 0.75, color: "white" });
    } else {
      viewer.setStyle({}, styleSpec(kind));
    }
    viewer.zoomTo();
    viewer.render();
    viewer.spin($("#spin").checked ? "y" : false);

    // quick facts straight from the file
    const atoms = (data.match(/^ATOM/gm) || []).length;
    const het = (data.match(/^HETATM/gm) || []).length;
    const chains = new Set((data.match(/^ATOM.{17}(.)/gm) || []).map(l => l.slice(-1)));
    const title = (data.match(/^TITLE\s+(.*)$/m) || [, ""])[1].trim();
    $("#info").innerHTML = [
      ["ID", id.toUpperCase()], ["atoms", atoms], ["hetero atoms", het],
      ["chains", chains.size || 1], ["source", db === "pdb" ? "RCSB PDB" : "AlphaFold DB"],
    ].map(([k, v]) => `<span class="chip">${k} <b>${v}</b></span>`).join("") +
      (title ? `<span class="chip">${title.slice(0, 70)}</span>` : "");
    st.textContent = "loaded ✓";
  } catch (e) {
    st.textContent = "";
    $("#exp-viewer").innerHTML = `<div class="viewer-error">⚠ ${e.message}<br><span>Check the ID (PDB: 4 chars like 1AKI · AlphaFold: a UniProt ID like P69905)</span></div>`;
  }
}

$("#examples").innerHTML = EXAMPLES.map((e, i) =>
  `<span class="pchip" data-i="${i}"><b>${e.label}</b> · ${e.id} <span class="muted">${e.note}</span></span>`).join("");
document.querySelectorAll(".pchip").forEach(c => c.onclick = () => {
  const e = EXAMPLES[+c.dataset.i];
  $("#db").value = e.db; $("#pid").value = e.id;
  load(e.db, e.id);
});
$("#load").onclick = () => load($("#db").value, $("#pid").value.trim());
$("#pid").addEventListener("keydown", ev => { if (ev.key === "Enter") $("#load").click(); });
$("#style").onchange = () => load($("#db").value, $("#pid").value.trim());
$("#spin").onchange = () => { if (viewer) viewer.spin($("#spin").checked ? "y" : false); };

load("pdb", "1AKI");   // land on something real immediately
