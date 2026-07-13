// Load a page, capture console + page errors + failed requests, probe DOM, screenshot.
// usage: node shot.js <url> <out.png> [waitMs]
const { chromium } = require('playwright');

(async () => {
  const url = process.argv[2];
  const out = process.argv[3] || 'shot.png';
  const wait = parseInt(process.argv[4] || '9000', 10);

  const browser = await chromium.launch({
    args: ['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'],
  });
  const page = await browser.newPage({ viewport: { width: 1500, height: 1100 } });

  const logs = [];
  page.on('console', m => logs.push(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', e => logs.push(`[PAGEERROR] ${e.message}\n  ${(e.stack || '').split('\n').slice(1, 4).join('\n  ')}`));
  page.on('requestfailed', r => logs.push(`[REQFAIL] ${r.url()} :: ${r.failure()?.errorText}`));
  page.on('response', r => { if (r.status() >= 400) logs.push(`[HTTP ${r.status()}] ${r.url()}`); });

  try { await page.goto(url, { waitUntil: 'networkidle', timeout: 45000 }); }
  catch (e) { logs.push(`[GOTO] ${e.message.split('\n')[0]}`); }

  await page.waitForTimeout(wait);

  const probe = await page.evaluate(() => {
    const q = s => document.querySelector(s);
    return {
      title: q('#run-title')?.textContent ?? null,
      plots_children: q('#plots') ? q('#plots').children.length : 'no #plots',
      viewer3dmol_hidden: q('#viewer')?.classList.contains('hidden') ?? 'missing',
      molstar_exists: !!q('#viewer-molstar'),
      molstar_hidden: q('#viewer-molstar')?.classList.contains('hidden') ?? 'missing',
      molstar_children: q('#viewer-molstar')?.children.length ?? 0,
      molstar_global: typeof window.molstar,
      MolstarView_global: typeof window.MolstarView,
      canvases: [...document.querySelectorAll('canvas')].map(c => `${c.width}x${c.height} in #${c.parentElement?.id || c.parentElement?.className}`),
    };
  });

  await page.screenshot({ path: out, fullPage: true });
  await browser.close();
  console.log('=== DOM PROBE ===\n' + JSON.stringify(probe, null, 2));
  console.log('\n=== CONSOLE / ERRORS ===\n' + (logs.length ? logs.join('\n') : '(none)'));
  console.log(`\nscreenshot -> ${out}`);
})();
