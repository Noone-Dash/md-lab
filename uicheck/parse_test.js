const BASE = process.env.MDLAB_URL || 'http://127.0.0.1:5057';
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch({ args:['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'] });
  const p = await b.newPage();
  await p.goto(BASE + '/explore', { waitUntil:'networkidle' }); // gives us $3Dmol
  const runs = {
    reaction: '/api/run/20260712_190819_451271_reaction_scan/traj',
    cell:     '/api/run/20260712_190840_850301_cell_rd/traj',
    openmm:   '/api/run/20260712_181038_openmm_implicit/traj',
  };
  for (const [name, url] of Object.entries(runs)) {
    const r = await p.evaluate(async (u) => {
      const txt = await (await fetch(u)).text();
      const div = document.createElement('div');
      div.style.width='400px'; div.style.height='300px'; div.style.position='relative';
      document.body.appendChild(div);
      const v = $3Dmol.createViewer(div, {});
      v.addModelsAsFrames(txt, 'pdb');
      const m = v.getModel(0);
      return {
        bytes: txt.length,
        models_in_text: (txt.match(/^MODEL/gm)||[]).length,
        atom_lines: (txt.match(/^ATOM/gm)||[]).length,
        parsed_atoms: m ? m.selectedAtoms({}).length : -1,
        frames: v.getNumFrames ? v.getNumFrames() : -1,
      };
    }, url);
    console.log(name.padEnd(9), JSON.stringify(r));
  }
  await b.close();
})();
