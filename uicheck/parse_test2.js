const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch({ args:['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'] });
  const p = await b.newPage();
  await p.goto('http://127.0.0.1:5057/explore', { waitUntil:'networkidle' });
  const runs = {
    reaction: ['20260712_204958_881842_reaction_scan', 3],
    cell:     ['20260712_204958_909363_cell_rd', 160],
    qmmm:     ['20260712_204958_923624_qmmm_opt', 4],
    openmm:   ['20260712_204958_942561_openmm_implicit', 138],
  };
  let fail = 0;
  for (const [name, [id, expect]] of Object.entries(runs)) {
    const r = await p.evaluate(async (u) => {
      const txt = await (await fetch(u)).text();
      const div = document.createElement('div');
      div.style.width='300px'; div.style.height='200px'; div.style.position='relative';
      document.body.appendChild(div);
      const v = $3Dmol.createViewer(div, {});
      v.addModelsAsFrames(txt, 'pdb');
      const m = v.getModel(0);
      return { atoms: m ? m.selectedAtoms({}).length : -1, frames: v.getNumFrames() };
    }, `/api/run/${id}/traj`);
    const ok = r.atoms === expect;
    if (!ok) fail++;
    console.log(`${ok?'OK  ':'FAIL'} ${name.padEnd(9)} parsed ${String(r.atoms).padStart(4)} / expected ${String(expect).padStart(4)}   frames=${r.frames}`);
  }
  await b.close();
  console.log(fail ? `\n${fail} STILL BROKEN` : '\nALL TRAJECTORIES PARSE CORRECTLY');
})();
