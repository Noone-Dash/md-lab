const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch({ args:['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'] });
  const p = await b.newPage();
  await p.goto('http://127.0.0.1:5057/explore', { waitUntil:'networkidle' });
  const opts = [
    ['default', null],
    ['{keepH:true}', {keepH:true}],
    ['{noH:false}', {noH:false}],
    ['{keepH:true,assignBonds:true}', {keepH:true, assignBonds:true}],
  ];
  for (const [label, o] of opts) {
    const r = await p.evaluate(async ([u, o]) => {
      const txt = await (await fetch(u)).text();
      const div = document.createElement('div');
      div.style.width='200px'; div.style.height='150px'; div.style.position='relative';
      document.body.appendChild(div);
      const v = $3Dmol.createViewer(div, {});
      if (o) v.addModelsAsFrames(txt, 'pdb', o); else v.addModelsAsFrames(txt, 'pdb');
      const m = v.getModel(0);
      const atoms = m ? m.selectedAtoms({}) : [];
      const elems = {};
      atoms.forEach(a => elems[a.elem] = (elems[a.elem]||0)+1);
      return { n: atoms.length, elems };
    }, ['/api/run/20260712_204958_881842_reaction_scan/traj', o]);
    console.log(`reaction (expect 3: O,H,H)  ${label.padEnd(30)} -> ${r.n}  ${JSON.stringify(r.elems)}`);
  }
  await b.close();
})();
