const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch({ args:['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'] });
  const p = await b.newPage({ viewport:{width:1500,height:1150} });
  const errs=[]; p.on('pageerror',e=>errs.push(e.message)); p.on('console',m=>{if(m.type()==='error')errs.push(m.text());});
  await p.goto('http://127.0.0.1:5057/track/classical?run=20260712_193742_689634_protein',{waitUntil:'networkidle'});
  await p.waitForTimeout(6000);
  // exercise the new controls: surface on, colour by secondary structure
  await p.selectOption('#surface','vdw');
  await p.selectOption('#color','ss');
  await p.waitForTimeout(6000);
  await p.screenshot({path:'v3d_protein_surface.png', fullPage:true});
  await b.close();
  console.log('errors:', errs.length? errs.join('\n') : 'none');
})();
