const BASE = process.env.MDLAB_URL || 'http://127.0.0.1:5057';
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch({ args:['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'] });
  const p = await b.newPage({ viewport:{width:1500,height:1200} });
  const errs=[]; p.on('pageerror',e=>errs.push(e.message)); p.on('console',m=>{if(m.type()==='error')errs.push(m.text());});
  await p.goto(BASE + '/plan',{waitUntil:'networkidle'});
  await p.waitForTimeout(3000);
  await p.screenshot({path:'plan_good.png', fullPage:true});
  // now click the deliberately broken example
  const btns = await p.$$('.ex-btn');
  await btns[2].click();
  await p.waitForTimeout(2500);
  await p.screenshot({path:'plan_broken.png', fullPage:true});
  const v = await p.$eval('#validation', e=>e.innerText.slice(0,400));
  const runBtn = await p.$eval('#btn-run', e=>e.textContent + ' disabled=' + e.disabled);
  await b.close();
  console.log('run button on broken plan:', runBtn);
  console.log('validation shown:\n', v);
  console.log('js errors:', errs.length?errs.join('\n'):'none');
})();
