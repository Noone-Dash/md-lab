const BASE = process.env.MDLAB_URL || 'http://127.0.0.1:5057';
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch({ args:['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'] });
  const p = await b.newPage();
  await p.goto(BASE + '/explore', { waitUntil:'networkidle' });
  // [id, expected atoms from the manifest]
  const runs = {
    reaction: ['20260712_204958_881842_reaction_scan', 3],
    qmmm:     ['20260712_204958_923624_qmmm_opt', 4],
    cell:     ['20260712_204958_909363_cell_rd', 160],
    openmm:   ['20260712_204958_942561_openmm_implicit', 138],
    protein:  ['20260712_193742_689634_protein', 1231],
    water:    ['20260712_185934_380270_water_box', null],
    lj:       ['20260712_202528_200693_lj_argon', null],
    martini:  ['20260712_175403_martini_bilayer', null],
  };
  let fail = 0;
  for (const [name, [id, expect]] of Object.entries(runs)) {
    const r = await p.evaluate(async ([u]) => {
      const txt = await (await fetch(u)).text();
      const div = document.createElement('div');
      div.style.width='200px'; div.style.height='150px'; div.style.position='relative';
      document.body.appendChild(div);
      const v = $3Dmol.createViewer(div, {});
      v.addModelsAsFrames(txt, 'pdb', { keepH: true });
      const m = v.getModel(0);
      const atoms = m ? m.selectedAtoms({}) : [];
      // ground truth: ATOM lines in the FIRST model
      const first = txt.split(/^ENDMDL/m)[0];
      const truth = (first.match(/^ATOM|^HETATM/gm) || []).length;
      const el = {}; atoms.forEach(a => el[a.elem] = (el[a.elem]||0)+1);
      return { parsed: atoms.length, truth, frames: v.getNumFrames(), el };
    }, [`/api/run/${id}/traj`]);
    const ok = r.parsed === r.truth && (expect === null || r.parsed === expect);
    if (!ok) fail++;
    console.log(`${ok?'OK  ':'FAIL'} ${name.padEnd(9)} parsed=${String(r.parsed).padStart(5)} / in-file=${String(r.truth).padStart(5)}  frames=${String(r.frames).padStart(3)}  ${JSON.stringify(r.el)}`);
  }
  await b.close();
  console.log(fail ? `\n${fail} STILL BROKEN` : '\nEVERY TRAJECTORY PARSES 100% OF ITS ATOMS');
})();
