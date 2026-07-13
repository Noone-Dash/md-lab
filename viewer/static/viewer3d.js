/* V3D — the molecular viewer: representations, colouring, selections, surfaces,
   labels, measurement, a legend, a live info HUD, and fullscreen.
   Owns the 3Dmol canvas + every control around it. */
window.V3D = (function () {
  const $ = s => document.querySelector(s);

  let viewer = null, model = null, stage = null;
  let nframes = 0, frame = 0, playing = false, timer = null, speed = 120;
  let times = [];                 // ps per frame, parsed from gmx TITLE lines
  let box = null;                 // {a,b,c} in Angstrom from CRYST1
  let surfId = null;
  let picked = [];                // atoms clicked, for distance measurement
  let showBox = false;
  const opts = { style: "auto", color: "auto", surface: "none", surfOpacity: 0.7,
                 labels: "none", category: "", caption: "",
                 hide: { water: false, ions: false, hydrogens: false } };

  const ELEM_COLORS = () => ($3Dmol.elementColors && $3Dmol.elementColors.defaultColors) || {};
  const hex = n => "#" + (n | 0).toString(16).padStart(6, "0");
  const WATER = ["SOL", "HOH", "WAT", "TIP3", "W"];
  const IONS = ["NA", "CL", "K", "MG", "CA", "ION", "NA+", "CL-"];

  /* ---------- lifecycle ---------- */
  function init(stageEl) {
    stage = stageEl;
    viewer = $3Dmol.createViewer(stageEl.querySelector("#viewer"), { backgroundColor: "#0a0d12" });
    window.addEventListener("resize", resize);
    document.addEventListener("fullscreenchange", () => setTimeout(resize, 120));
    wire();
    return viewer;
  }

  function parseTimes(pdb) {
    const t = []; const re = /^TITLE.*?t=\s*([0-9.eE+-]+)/gm; let m;
    while ((m = re.exec(pdb))) t.push(parseFloat(m[1]));
    return t;
  }
  function parseBox(pdb) {
    const m = /^CRYST1\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)/m.exec(pdb);
    return m ? { a: +m[1], b: +m[2], c: +m[3] } : null;
  }

  function load(pdb, meta) {
    Object.assign(opts, meta || {});
    viewer.clear(); surfId = null; picked = []; viewer.removeAllLabels();
    // keepH: 3Dmol's PDB parser DISCARDS every hydrogen by default — without this
    // a water molecule renders as a lone oxygen and a protein loses ~half its atoms.
    viewer.addModelsAsFrames(pdb, "pdb", { keepH: true });
    model = viewer.getModel(0);
    nframes = viewer.getNumFrames ? viewer.getNumFrames() : 1;
    times = parseTimes(pdb);
    box = parseBox(pdb);
    frame = 0;

    buildLegend();
    buildInfo();
    $("#vcaption").innerHTML = opts.caption
      ? `<b>What you're looking at</b> ${opts.caption}` : "";
    $("#vcaption").style.display = opts.caption ? "" : "none";

    applyStyle();
    viewer.zoomTo();
    viewer.render();

    const sl = $("#frame"); sl.max = Math.max(0, nframes - 1); sl.value = 0;
    updateHUD();
    if (nframes > 1) play();
  }

  /* ---------- representation ---------- */
  function autoKind() {
    if (/Biomol/i.test(opts.category)) return "cartoon";
    if (/Quantum/i.test(opts.category)) return "ballstick";
    return "sphere";
  }
  function bead() { return /Cell|Coarse/i.test(opts.category); }

  function colorFor(kind) {
    const c = opts.color === "auto"
      ? (kind === "cartoon" ? "spectrum" : "element") : opts.color;
    switch (c) {
      case "element": return { generic: {}, cartoon: { colorscheme: "default" } };
      case "spectrum": return { generic: { colorscheme: { prop: "resi", gradient: "roygb" } },
                                cartoon: { color: "spectrum" } };
      case "ss": return { generic: {}, cartoon: { colorscheme: "ssPyMOL" } };
      case "bfactor": return { generic: { colorscheme: { prop: "b", gradient: "rwb" } },
                               cartoon: { colorscheme: { prop: "b", gradient: "rwb" } } };
      default: return { generic: {}, cartoon: { color: "spectrum" } };
    }
  }

  function styleSpec() {
    const kind = opts.style === "auto" ? autoKind() : opts.style;
    const c = colorFor(kind);
    switch (kind) {
      case "cartoon": return { cartoon: { ...c.cartoon }, stick: { radius: 0.08, ...c.generic } };
      case "stick": return { stick: { radius: 0.15, ...c.generic } };
      case "ballstick": return { stick: { radius: 0.12, ...c.generic }, sphere: { scale: 0.25, ...c.generic } };
      case "line": return { line: { ...c.generic } };
      case "sphere":
      default: return { sphere: { scale: bead() ? 0.5 : 0.32, ...c.generic } };
    }
  }

  function applyStyle() {
    if (!viewer) return;
    viewer.setStyle({}, {});                       // clear everything
    viewer.setStyle({}, styleSpec());              // then paint visible
    if (opts.hide.water) viewer.setStyle({ resn: WATER }, {});
    if (opts.hide.ions) viewer.setStyle({ resn: IONS }, {});
    if (opts.hide.hydrogens) viewer.setStyle({ elem: "H" }, {});
    applySurface();
    applyLabels();
    applyBox();
    viewer.render();
  }

  function applySurface() {
    try { if (surfId !== null) { viewer.removeSurface(surfId); surfId = null; } } catch (e) {}
    if (opts.surface === "none") return;
    const t = opts.surface === "sas" ? $3Dmol.SurfaceType.SAS : $3Dmol.SurfaceType.VDW;
    const sel = opts.hide.water ? { not: { resn: WATER } } : {};
    const r = viewer.addSurface(t, { opacity: +opts.surfOpacity, color: "#8fd0ff" }, sel);
    if (r && typeof r.then === "function") r.then(id => { surfId = id; viewer.render(); });
    else surfId = r;
  }

  function applyBox() {
    try {
      if (showBox && model) viewer.addUnitCell(model, { box: { color: "#4f9dff" } });
      else if (model) viewer.removeUnitCell(model);
    } catch (e) {}
  }

  function applyLabels() {
    viewer.removeAllLabels();
    if (opts.labels === "residues" && model) {
      const seen = new Set();
      model.selectedAtoms({}).forEach(a => {
        const key = a.chain + ":" + a.resi;
        if (seen.has(key) || (a.atom !== "CA" && a.atom !== "PO4" && a.atom !== "OW")) return;
        seen.add(key);
        viewer.addLabel(`${a.resn}${a.resi}`, {
          position: { x: a.x, y: a.y, z: a.z }, fontSize: 10,
          fontColor: "white", backgroundColor: "#11161f", backgroundOpacity: 0.6 });
      });
    }
    // re-label the measurement picks
    picked.forEach(a => viewer.addLabel(`${a.resn}${a.resi || ""} ${a.atom}`, {
      position: { x: a.x, y: a.y, z: a.z }, fontSize: 11, fontColor: "#33d9a6",
      backgroundColor: "#0b0e13", backgroundOpacity: 0.8 }));
  }

  /* ---------- legend + info ---------- */
  function atomList() { return model ? model.selectedAtoms({}) : []; }

  function buildLegend() {
    const a = atomList();
    const byElem = {}, byRes = {};
    a.forEach(x => {
      byElem[x.elem || "?"] = (byElem[x.elem || "?"] || 0) + 1;
      byRes[x.resn || "?"] = (byRes[x.resn || "?"] || 0) + 1;
    });
    const ec = ELEM_COLORS();
    const elems = Object.entries(byElem).sort((p, q) => q[1] - p[1]).slice(0, 8);
    const res = Object.entries(byRes).sort((p, q) => q[1] - p[1]).slice(0, 6);
    $("#vlegend").innerHTML =
      `<div class="lg-title">Legend</div>` +
      elems.map(([e, n]) =>
        `<div class="lg-row"><i style="background:${hex(ec[e] ?? 0x909090)}"></i>
           <span>${e}</span><b>${n.toLocaleString()}</b></div>`).join("") +
      (res.length > 1 ? `<div class="lg-title" style="margin-top:8px">Residues</div>` +
        res.map(([r, n]) => `<div class="lg-row"><span>${r}</span><b>${n.toLocaleString()}</b></div>`).join("") : "");
  }

  function buildInfo() {
    const n = atomList().length;
    const rows = [["atoms", n.toLocaleString()], ["frames", nframes]];
    if (box) rows.push(["box", `${(box.a / 10).toFixed(1)} × ${(box.b / 10).toFixed(1)} × ${(box.c / 10).toFixed(1)} nm`]);
    if (times.length > 1) rows.push(["duration", fmtTime(times[times.length - 1])]);
    $("#vinfo").innerHTML = rows.map(([k, v]) =>
      `<div class="lg-row"><span>${k}</span><b>${v}</b></div>`).join("") +
      `<div class="lg-row" id="vtime"></div>`;
  }

  function fmtTime(ps) {
    if (ps == null) return "—";
    return ps >= 1000 ? (ps / 1000).toFixed(2) + " ns" : ps.toFixed(1) + " ps";
  }

  /* ---------- playback ---------- */
  function showFrame(i) {
    frame = i;
    if (viewer.setFrame) viewer.setFrame(i).then(() => viewer.render());
    updateHUD();
  }
  function updateHUD() {
    $("#frame-label").textContent = `${frame + 1} / ${nframes}`;
    $("#frame").value = frame;
    const t = $("#vtime");
    if (t) t.innerHTML = times.length > frame
      ? `<span>time</span><b>${fmtTime(times[frame])}</b>` : "";
  }
  function play() {
    if (nframes < 2) return;
    playing = true; $("#play").textContent = "❚❚";
    clearInterval(timer);
    timer = setInterval(() => showFrame((frame + 1) % nframes), speed);
  }
  function pause() { playing = false; $("#play").textContent = "▶"; clearInterval(timer); }

  /* ---------- view ---------- */
  function resize() { try { viewer && viewer.resize(); } catch (e) {} }
  function resetView() { viewer.zoomTo(); viewer.render(); }
  function zoom(f) { viewer.zoom(f, 300); }
  function fullscreen() {
    if (!document.fullscreenElement) stage.requestFullscreen?.();
    else document.exitFullscreen?.();
  }

  /* ---------- picking / measuring ---------- */
  function enablePicking() {
    viewer.setClickable({}, true, atom => {
      picked.push(atom);
      if (picked.length > 2) picked = [atom];
      if (picked.length === 2) {
        const [p, q] = picked;
        const d = Math.hypot(p.x - q.x, p.y - q.y, p.z - q.z);
        viewer.addLabel(`${d.toFixed(2)} Å`, {
          position: { x: (p.x + q.x) / 2, y: (p.y + q.y) / 2, z: (p.z + q.z) / 2 },
          fontSize: 12, fontColor: "#ffb454", backgroundColor: "#0b0e13", backgroundOpacity: 0.85 });
        viewer.addLine({ start: { x: p.x, y: p.y, z: p.z }, end: { x: q.x, y: q.y, z: q.z },
                         color: "#ffb454", dashed: true });
        $("#measure-out").textContent = `${p.resn}${p.resi || ""}:${p.atom} → ${q.resn}${q.resi || ""}:${q.atom} = ${d.toFixed(2)} Å`;
      } else {
        $("#measure-out").textContent = `picked ${atom.resn}${atom.resi || ""}:${atom.atom} — click another atom to measure`;
      }
      applyLabels(); viewer.render();
    });
  }
  function clearPicks() {
    picked = []; $("#measure-out").textContent = "";
    viewer.removeAllLabels(); viewer.removeAllShapes(); applyStyle();
  }

  /* ---------- wire the controls ---------- */
  function wire() {
    $("#play").onclick = () => (playing ? pause() : play());
    $("#frame").oninput = e => { pause(); showFrame(+e.target.value); };
    $("#speed").onchange = e => { speed = +e.target.value; if (playing) play(); };

    $("#style").onchange = e => { opts.style = e.target.value; applyStyle(); };
    $("#color").onchange = e => { opts.color = e.target.value; applyStyle(); };
    $("#surface").onchange = e => { opts.surface = e.target.value; applyStyle(); };
    $("#surfop").oninput = e => { opts.surfOpacity = e.target.value; applyStyle(); };
    $("#labels").onchange = e => { opts.labels = e.target.value; applyStyle(); };

    $("#hide-water").onchange = e => { opts.hide.water = e.target.checked; applyStyle(); };
    $("#hide-ions").onchange = e => { opts.hide.ions = e.target.checked; applyStyle(); };
    $("#hide-h").onchange = e => { opts.hide.hydrogens = e.target.checked; applyStyle(); };
    $("#show-box").onchange = e => { showBox = e.target.checked; applyStyle(); };
    $("#spin").onchange = e => viewer.spin(e.target.checked ? "y" : false);

    $("#zoom-in").onclick = () => zoom(1.25);
    $("#zoom-out").onclick = () => zoom(0.8);
    $("#reset-view").onclick = resetView;
    $("#fs").onclick = fullscreen;
    $("#clear-picks").onclick = clearPicks;
    enablePicking();
  }

  return { init, load, resize, pause };
})();
