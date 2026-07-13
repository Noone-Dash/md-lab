const BASE = process.env.MDLAB_URL || 'http://127.0.0.1:5057';
// Drive the UI like a human: pick an experiment, tweak a param, click Launch,
// wait for the run to finish, screenshot. Proof that submit actually works.
const { chromium } = require('playwright');

(async () => {
  const base = process.argv[2] || BASE;
  const browser = await chromium.launch({
    args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
  });
  const page = await browser.newPage({ viewport: { width: 1500, height: 1100 } });
  const errs = [];
  page.on('pageerror', e => errs.push('[PAGEERROR] ' + e.message));
  page.on('console', m => { if (m.type() === 'error') errs.push('[error] ' + m.text()); });

  await page.goto(base + '/track/classical', { waitUntil: 'networkidle' });

  // click the first experiment card (Lennard-Jones argon)
  await page.waitForSelector('.rcard');
  const cardName = await page.$eval('.rcard .rname', e => e.textContent);
  await page.click('.rcard');
  await page.waitForSelector('#param-fields .pfield');

  // how many knobs does the UI actually give us?
  const knobs = await page.$$eval('#param-fields [data-name]', els =>
    els.map(e => e.dataset.name));

  // tweak temperature slider if present, then launch
  const temp = await page.$('#p_temperature');
  if (temp) await page.$eval('#p_temperature', el => { el.value = 90; el.dispatchEvent(new Event('input')); });
  await page.$eval('#p_nsteps', el => { el.value = 3000; el.dispatchEvent(new Event('input')); }).catch(() => {});

  await page.click('.launch');

  // wait for a terminal status
  let status = '';
  for (let i = 0; i < 60; i++) {
    await page.waitForTimeout(2000);
    status = await page.$eval('#run-status', e => e.textContent.trim()).catch(() => '');
    if (/done|error/i.test(status)) break;
  }
  const plots = await page.$$eval('#plots .plot-card', e => e.length).catch(() => 0);
  const frames = await page.$eval('#frame-label', e => e.textContent).catch(() => 'n/a');

  await page.screenshot({ path: 'submit_test.png', fullPage: true });
  await browser.close();

  console.log('experiment clicked : ' + cardName);
  console.log('knobs exposed in UI: ' + knobs.length + '  -> ' + knobs.join(', '));
  console.log('final status       : ' + status);
  console.log('plots rendered     : ' + plots);
  console.log('trajectory frames  : ' + frames);
  console.log('js errors          : ' + (errs.length ? errs.join('\n  ') : 'none'));
})();
