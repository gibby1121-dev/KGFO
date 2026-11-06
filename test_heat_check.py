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
