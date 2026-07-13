# uicheck — look at the page before claiming it works

    cd uicheck && npm install playwright && npx playwright install chromium
    node shot.js "http://127.0.0.1:5057/explore" out.png 9000

Prints the DOM probe + every console error/pageerror/failed request, and writes a
full-page screenshot. Chromium needs ARM64-safe GL flags (already in shot.js):
`--use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader`.
(Puppeteer's bundled Chrome is broken on this ARM box — use Playwright.)
