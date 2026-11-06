"""Offline check of global_heat_check.py using synthetic prices (no network)."""
import io
import contextlib

import numpy as np
import pandas as pd

import global_heat_check as g

idx = pd.bdate_range(end="2026-11-04", periods=800)
rng = np.random.default_rng(1)
cols = sorted(g.G20_MARKETS)
px = pd.DataFrame(
    100 * np.exp(np.cumsum(rng.normal(0, 0.01, (800, len(cols))), axis=0)),
    index=idx, columns=cols,
)
px.iloc[-1, cols.index("ARGT")] *= 1.12  # forced breakout

_real_fetch = g.fetch_prices
g.fetch_prices = lambda tickers, window: px

for fmt in ("markdown", "text"):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert g.main(["--format", fmt]) == 0
    out = buf.getvalue()
    assert "G7 Markets Heat Check" in out and "G20 Markets Heat Check" in out
    assert "COMPOSITE" in out and "commentary" in out
    assert "ARGT" in out and "PRICE+" in out
    assert "Composite excludes: FEZ" in out

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g.main(["--no-commentary"])
assert "commentary" not in buf.getvalue()
print("ok")

import os
import tempfile
with tempfile.TemporaryDirectory() as d:
    with contextlib.redirect_stdout(io.StringIO()):
        g.main(["--dashboard-dir", d])
    for fn, must in (("g7_dashboard.html", "G7 Markets Heat Check"), ("g20_dashboard.html", "G20 Markets Heat Check")):
        page = open(os.path.join(d, fn), encoding="utf-8").read()
        assert must in page and "COMPOSITE" in page and "<svg" in page
print("dashboards ok")

# Trailing placeholder row: one ticker missing the newest session must be dropped,
# not forward-filled into a fake 0% day.
import types
cols2 = ["SPY", "EWJ"]
idx2 = pd.bdate_range(end="2026-11-05", periods=5)
frame = pd.DataFrame({"SPY": [100, 101, 102, 103, 104.0], "EWJ": [50, 51, 52, 53, np.nan]}, index=idx2)
raw = pd.concat({"Close": frame}, axis=1)
g.yf = types.SimpleNamespace(download=lambda *a, **k: raw)
out = g._drop_incomplete_trailing_rows(frame)
assert out.index[-1] == idx2[-2] and len(out) == 4
full = _real_fetch(cols2, 252)
assert full.index[-1] == idx2[-2] and full["EWJ"].iloc[-1] == 53
# A never-listed ticker (all NaN) must not trigger dropping.
frame2 = frame.assign(EWJ=[np.nan] * 5)
assert len(g._drop_incomplete_trailing_rows(frame2.dropna(how="all"))) == 5
print("trailing-row ok")
