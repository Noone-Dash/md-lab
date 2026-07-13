"""Parse GROMACS ``.xvg`` (grace/xmgrace) files into plain dicts for the UI."""

from __future__ import annotations

from pathlib import Path


def parse_xvg(path) -> dict:
    """Return {title, xaxis, yaxis, legends, x:[...], series:[[...], ...]}.

    Handles the ``@`` metadata lines (title/axis/legend) and the numeric block.
    """
    path = Path(path)
    title = ""
    xaxis = ""
    yaxis = ""
    legends: list[str] = []
    x: list[float] = []
    cols: list[list[float]] = []

    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                continue
            if line.startswith("@"):
                low = line.lower()
                if "title" in low and '"' in line:
                    title = line.split('"', 1)[1].rsplit('"', 1)[0]
                elif "xaxis" in low and "label" in low and '"' in line:
                    xaxis = line.split('"', 1)[1].rsplit('"', 1)[0]
                elif "yaxis" in low and "label" in low and '"' in line:
                    yaxis = line.split('"', 1)[1].rsplit('"', 1)[0]
                elif low.strip().startswith("@ s") and "legend" in low and '"' in line:
                    legends.append(line.split('"', 1)[1].rsplit('"', 1)[0])
                continue
            parts = line.split()
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if not vals:
                continue
            x.append(vals[0])
            ys = vals[1:]
            if not cols:
                cols = [[] for _ in ys]
            for i, v in enumerate(ys):
                if i < len(cols):
                    cols[i].append(v)

    return {
        "title": title,
        "xaxis": xaxis,
        "yaxis": yaxis,
        "legends": legends,
        "x": x,
        "series": cols,
    }
