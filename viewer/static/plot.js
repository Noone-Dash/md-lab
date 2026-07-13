/* Tiny dependency-free multi-series line plotter on <canvas>. */
(function () {
  const PALETTE = ["#4f9dff", "#33d9a6", "#ffb454", "#ff6b9d", "#b48cff", "#5fd0e6", "#ff8f5f"];

  function niceTicks(min, max, count) {
    if (min === max) { min -= 1; max += 1; }
    const span = max - min;
    const step0 = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const norm = step0 / mag;
    const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
    const start = Math.ceil(min / step) * step;
    const ticks = [];
    for (let v = start; v <= max + step * 1e-6; v += step) ticks.push(v);
    return ticks;
  }

  function fmt(v) {
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-2 || a >= 1e5)) return v.toExponential(1);
    if (a >= 1000) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    return v.toFixed(2);
  }

  window.drawLinePlot = function (canvas, opts) {
    const { x, series, xlabel, ylabel } = opts;
    const colors = opts.colors || PALETTE;
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 320;
    const cssH = canvas.clientHeight || 180;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const padL = 46, padR = 10, padT = 10, padB = 26;
    const W = cssW - padL - padR, H = cssH - padT - padB;

    let xmin = Math.min(...x), xmax = Math.max(...x);
    let ymin = Infinity, ymax = -Infinity;
    for (const s of series) for (const v of s) {
      if (v < ymin) ymin = v; if (v > ymax) ymax = v;
    }
    if (!isFinite(ymin)) { ymin = 0; ymax = 1; }
    const yPad = (ymax - ymin) * 0.08 || 1;
    ymin -= yPad; ymax += yPad;

    const X = v => padL + (v - xmin) / (xmax - xmin || 1) * W;
    const Y = v => padT + H - (v - ymin) / (ymax - ymin || 1) * H;

    // grid + ticks
    ctx.strokeStyle = "#26303f"; ctx.fillStyle = "#8593a6";
    ctx.lineWidth = 1; ctx.font = "10px system-ui"; ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    for (const t of niceTicks(ymin, ymax, 4)) {
      const y = Y(t);
      ctx.globalAlpha = 0.5; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + W, y); ctx.stroke();
      ctx.globalAlpha = 1; ctx.fillText(fmt(t), padL - 6, y);
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const t of niceTicks(xmin, xmax, 5)) {
      const xp = X(t);
      if (xp < padL - 1 || xp > padL + W + 1) continue;
      ctx.globalAlpha = 0.25; ctx.beginPath(); ctx.moveTo(xp, padT); ctx.lineTo(xp, padT + H); ctx.stroke();
      ctx.globalAlpha = 1; ctx.fillText(fmt(t), xp, padT + H + 6);
    }

    // axis labels
    ctx.fillStyle = "#8593a6"; ctx.globalAlpha = 1;
    if (xlabel) { ctx.textAlign = "right"; ctx.fillText(xlabel, padL + W, padT + H + 15); }
    if (ylabel) {
      ctx.save(); ctx.translate(11, padT + H / 2); ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center"; ctx.fillText(ylabel, 0, 0); ctx.restore();
    }

    // series
    ctx.lineWidth = 1.8; ctx.lineJoin = "round";
    series.forEach((s, si) => {
      ctx.strokeStyle = colors[si % colors.length];
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < s.length; i++) {
        const px = X(x[i]), py = Y(s[i]);
        if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
      }
      ctx.stroke();
    });
  };

  window.PLOT_PALETTE = PALETTE;
})();
