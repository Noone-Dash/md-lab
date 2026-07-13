// Full sweep: every page + one run per track. Reports console errors and blank canvases.
const { chromium } = require('playwright');

const PAGES = [
  ['hub', '/'],
  ['explore', '/explore'],
  ['monitor', '/monitor'],
  ['track-classical', '/track/classical'],
  ['track-cg', '/track/cg'],
  ['track-openmm', '/track/openmm'],
  ['track-qmmm', '/track/qmmm'],
  ['track-cell', '/track/cell'],
  ['run-lj', '/track/classical?run=20260712_202528_200693_lj_argon'],
  ['run-protein', '/track/classical?run=20260712_193742_689634_protein'],
  ['run-martini', '/track/cg?run=20260712_175403_martini_bilayer'],
  ['run-openmm', '/track/openmm?run=20260712_204958_942561_openmm_implicit'],
  ['run-reaction', '/track/qmmm?run=20260712_204958_881842_reaction_scan'],
  ['run-cell', '/track/cell?run=20260712_204958_909363_cell_rd'],
];

(async () => {
  const base = 'http://127.0.0.1:5057';
  const browser = await chromium.launch({
    args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
  });
  let bad = 0;
  for (const [name, path] of PAGES) {
    const page = await browser.newPage({ viewport: { width: 1500, height: 1100 } });
    const errs = [];
    page.on('pageerror', e => errs.push('PAGEERROR: ' + e.message));
    page.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text().slice(0, 120)); });
    page.on('response', r => { if (r.status() >= 400) errs.push(`HTTP ${r.status()} ${r.url().replace(base,'')}`); });

    try { await page.goto(base + path, { waitUntil: 'networkidle', timeout: 40000 }); }
    catch (e) { errs.push('GOTO: ' + e.message.split('\n')[0]); }
    await page.waitForTimeout(path.includes('run=') ? 8000 : 3500);

    // is the 3D canvas actually painted (not an empty black box)?
    let painted = 'n/a';
    if (path.includes('run=')) {
      painted = await page.evaluate(() => {
        const c = document.querySelector('#viewer canvas');
        if (!c) return 'NO CANVAS';
        const g = document.createElement('canvas');
        g.width = c.width; g.height = c.height;
        try {
          g.getContext('2d').drawImage(c, 0, 0);
          const d = g.getContext('2d').getImageData(0, 0, g.width, g.height).data;
          let lit = 0;
          for (let i = 0; i < d.length; i += 400) if (d[i] > 25 || d[i+1] > 25 || d[i+2] > 25) lit++;
          return lit > 20 ? 'PAINTED (' + lit + ' lit px)' : 'BLANK/BLACK (' + lit + ')';
        } catch (e) { return 'readback failed: ' + e.message; }
      });
    }
    const plots = await page.$$eval('#plots .plot-card', e => e.length).catch(() => 'n/a');

    await page.screenshot({ path: `sweep_${name}.png`, fullPage: true });
    await page.close();

    const ok = errs.length === 0 && !String(painted).startsWith('BLANK') && painted !== 'NO CANVAS';
    if (!ok) bad++;
    console.log(`${ok ? 'OK  ' : 'FAIL'} ${name.padEnd(16)} canvas=${String(painted).padEnd(22)} plots=${plots}`);
    errs.slice(0, 3).forEach(e => console.log('       ! ' + e));
  }
  await browser.close();
  console.log(`\n${bad === 0 ? 'ALL PAGES CLEAN' : bad + ' PAGE(S) WITH PROBLEMS'}`);
})();
