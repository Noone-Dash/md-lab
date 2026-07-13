"""Metric extractors — pull a single number out of a run manifest so it can be
compared against a physical reference value."""

from __future__ import annotations


def _series(manifest, name):
    """Find a series by energy-term name or analysis name."""
    e = manifest.get("energy")
    if e:
        for lg, ser in zip(e["legends"], e["series"]):
            if lg.replace(" ", "").replace(".", "").lower() == name.replace(" ", "").lower():
                return e["x"], ser
    for a in manifest.get("analyses", []):
        if a["name"] == name:
            d = a["data"]
            return d["x"], (d["series"][0] if d["series"] else [])
    return None, None


def energy_mean(manifest, term, last_frac=0.5, **_):
    x, y = _series(manifest, term)
    if not y:
        return None
    k = max(1, int(len(y) * last_frac))
    tail = y[-k:]
    return sum(tail) / len(tail)


def analysis_final(manifest, name, last_frac=0.25, **_):
    x, y = _series(manifest, name)
    if not y:
        return None
    k = max(1, int(len(y) * last_frac))
    tail = y[-k:]
    return sum(tail) / len(tail)


def rdf_peak(manifest, which="pos", name="rdf", smooth=5, **_):
    """Tallest peak of g(r): its position (nm) or height.

    g(r) is a histogram: at low density individual bins are noisy and a SINGLE bin
    can spike (a 2.19 spike in an otherwise flat g(r)=1.0 gas made this report a
    structured liquid). Smooth over a few bins before taking the max — that is what
    the peak of a distribution actually means.
    """
    x, y = _series(manifest, name)
    if not y:
        return None
    pts = [(xi, yi) for xi, yi in zip(x, y) if xi > 0.15]
    if len(pts) < smooth:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    h = smooth // 2
    sm = []
    for i in range(len(ys)):
        lo, hi = max(0, i - h), min(len(ys), i + h + 1)
        w = ys[lo:hi]
        sm.append(sum(w) / len(w))
    i = max(range(len(sm)), key=lambda k: sm[k])
    return xs[i] if which == "pos" else sm[i]


def summary_value(manifest, key, **_):
    """Pull a number out of the run's key/value summary (QM runs)."""
    for k, v in manifest.get("summary", []) or []:
        if k.lower().startswith(key.lower()):
            num = ""
            for ch in str(v):
                if ch in "-+.0123456789eE":
                    num += ch
                elif num:
                    break
            try:
                return float(num)
            except ValueError:
                return None
    return None


def drift(manifest, term, **_):
    """How much a quantity moved between the first and last quarter — a stability check."""
    x, y = _series(manifest, term)
    if not y or len(y) < 4:
        return None
    k = max(1, len(y) // 4)
    return abs(sum(y[-k:]) / k - sum(y[:k]) / k)


EXTRACTORS = {
    "energy_mean": energy_mean,
    "analysis_final": analysis_final,
    "rdf_peak": rdf_peak,
    "summary": summary_value,
    "drift": drift,
}


def extract(manifest, spec: dict):
    fn = EXTRACTORS.get(spec.get("type"))
    if not fn:
        return None
    kw = {k: v for k, v in spec.items() if k != "type"}
    try:
        return fn(manifest, **kw)
    except Exception:  # noqa: BLE001
        return None
