#!/usr/bin/env python3
"""
Global Market Heat Check -- G7 / G20 equity proxy tracker.

Builds country-level equity proxies from flagship ETFs (via yfinance), computes
multi-horizon returns, equal-weight (or user-weighted) cluster baselines, and
50-day z-score / momentum-z breakout flags. Output is tables only.

Usage:
    pip install yfinance pandas numpy
    python global_heat_check.py [--window 252] [--z 2.0] [--format markdown|text]

Notes / limitations (by construction, not commentary):
  * Russia (G20) has no investable USD ETF on Yahoo; it is omitted.
  * FEZ (Eurozone) overlaps DE/FR/IT. It is listed in G20 but excluded from the
    G20 composite by default (COMPOSITE_EXCLUDE) to avoid double counting.
  * Prices are USD-denominated ETF prices, so returns include FX effects.
  * Market-cap weights are not available from Yahoo in a stable way; supply your
    own via CUSTOM_WEIGHTS (ticker -> weight). Empty => equal weight.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

# --------------------------------------------------------------------------- #
# 1. Universe mapping
# --------------------------------------------------------------------------- #
# ticker -> (country, asset name)
G7_MARKETS: dict[str, tuple[str, str]] = {
    "SPY": ("United States", "SPDR S&P 500 ETF"),
    "EWJ": ("Japan", "iShares MSCI Japan ETF"),
    "EWG": ("Germany", "iShares MSCI Germany ETF"),
    "EWU": ("United Kingdom", "iShares MSCI United Kingdom ETF"),
    "EWQ": ("France", "iShares MSCI France ETF"),
    "EWC": ("Canada", "iShares MSCI Canada ETF"),
    "EWI": ("Italy", "iShares MSCI Italy ETF"),
}

G20_MARKETS: dict[str, tuple[str, str]] = {
    **G7_MARKETS,
    "FEZ": ("Eurozone (EU proxy)", "SPDR Euro STOXX 50 ETF"),
    "EPI": ("India", "WisdomTree India Earnings ETF"),
    "FXI": ("China", "iShares China Large-Cap ETF"),
    "EWZ": ("Brazil", "iShares MSCI Brazil ETF"),
    "EWA": ("Australia", "iShares MSCI Australia ETF"),
    "EWY": ("South Korea", "iShares MSCI South Korea ETF"),
    "EWW": ("Mexico", "iShares MSCI Mexico ETF"),
    "EZA": ("South Africa", "iShares MSCI South Africa ETF"),
    "KSA": ("Saudi Arabia", "iShares MSCI Saudi Arabia ETF"),
    "TUR": ("Turkey", "iShares MSCI Turkey ETF"),
    "EIDO": ("Indonesia", "iShares MSCI Indonesia ETF"),
    "ARGT": ("Argentina", "Global X MSCI Argentina ETF"),
}

MARKET_GROUPS = {"G7_Markets": G7_MARKETS, "G20_Markets": G20_MARKETS}

COMPOSITE_EXCLUDE: dict[str, set[str]] = {"G7_Markets": set(), "G20_Markets": {"FEZ"}}
CUSTOM_WEIGHTS: dict[str, float] = {}  # e.g. {"SPY": 0.6, "EWJ": 0.1, ...}; empty => equal weight

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
HORIZONS = {"1D": 1, "5D": 5, "1M": 21, "3M": 63}  # trading days; YTD handled separately
MA_WINDOW = 50
MOM_LOOKBACK = 5  # days for momentum-z


@dataclass(frozen=True)
class Config:
    window: int = 252
    z_threshold: float = 2.0
    fmt: str = "markdown"


# --------------------------------------------------------------------------- #
# Data ingestion
# --------------------------------------------------------------------------- #
def fetch_prices(tickers: list[str], window: int) -> pd.DataFrame:
    """Daily adjusted closes for the last `window` trading days.

    Downloads extra history so the 50-day statistics are fully populated at the
    start of the window, then the frame is trimmed to `window` rows.
    """
    raw = yf.download(
        tickers, period="3y", interval="1d", auto_adjust=True,
        progress=False, group_by="column", threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data (network or ticker issue).")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = _drop_incomplete_trailing_rows(close.dropna(how="all")).ffill()
    return close


def _drop_incomplete_trailing_rows(close: pd.DataFrame) -> pd.DataFrame:
    """Drop trailing rows where a ticker that traded the prior day has no close.

    Yahoo can return a placeholder bar for the current session before every
    ETF has a close. Forward-filling it would stamp the report with a date that
    has no real data and show a 0% 1D return for the missing tickers.
    Tickers with no data at all (never listed/failed download) do not count.
    """
    while len(close) > 1:
        last, prev = close.iloc[-1], close.iloc[-2]
        stale = last.isna() & prev.notna()
        if not stale.any():
            break
        print(f"Note: dropped incomplete row {close.index[-1].date()} "
              f"(no close for: {', '.join(stale[stale].index)})", file=sys.stderr)
        close = close.iloc[:-1]
    return close


# --------------------------------------------------------------------------- #
# Analytics
# --------------------------------------------------------------------------- #
def trailing_returns(px: pd.Series, window_px: pd.Series) -> dict[str, float]:
    """Percent returns over fixed horizons + YTD, from a price series.

    `px` is the full history (for YTD base lookup); `window_px` is the trimmed window.
    """
    out: dict[str, float] = {}
    last = px.iloc[-1]
    for name, n in HORIZONS.items():
        out[name] = (last / px.iloc[-1 - n] - 1) * 100 if len(px) > n else np.nan
    year = px.index[-1].year
    prior = px[px.index.year < year]
    base = prior.iloc[-1] if not prior.empty else window_px.iloc[0]
    out["YTD"] = (last / base - 1) * 100
    return out


def breakout_stats(px: pd.Series, z_thr: float, window: int) -> dict[str, float | str]:
    """Price z-score vs 50d MA and 5-day momentum z-score.

    price_z = (P - SMA50) / rolling_std50(P)
    mom_z   = (R5_now - mean(R5 over window)) / std(R5 over window)
    Flag fires if |price_z| >= z_thr or |mom_z| >= z_thr.
    """
    sma = px.rolling(MA_WINDOW).mean()
    sd = px.rolling(MA_WINDOW).std(ddof=1)
    price_z = ((px - sma) / sd).iloc[-1]

    r5 = px.pct_change(MOM_LOOKBACK).dropna().iloc[-window:]
    mom_z = (r5.iloc[-1] - r5.mean()) / r5.std(ddof=1) if len(r5) > 10 else np.nan

    flags = []
    if pd.notna(price_z) and abs(price_z) >= z_thr:
        flags.append("PRICE" + ("+" if price_z > 0 else "-"))
    if pd.notna(mom_z) and abs(mom_z) >= z_thr:
        flags.append("MOM" + ("+" if mom_z > 0 else "-"))
    return {"Z50": price_z, "MomZ": mom_z, "Flag": ",".join(flags) if flags else "-"}


def composite_index(prices: pd.DataFrame, weights: dict[str, float] | None) -> pd.Series:
    """Daily-rebalanced composite (equal weight unless weights supplied)."""
    rets = prices.pct_change().dropna(how="all")
    if weights:
        w = pd.Series({t: weights.get(t, 0.0) for t in rets.columns})
        w = w[w > 0]
        rets = rets[w.index]
        # Renormalize daily over available constituents.
        mask = rets.notna()
        wmat = mask.mul(w, axis=1)
        port = (rets.fillna(0) * wmat).sum(axis=1) / wmat.sum(axis=1)
    else:
        port = rets.mean(axis=1, skipna=True)
    return (1 + port).cumprod() * 100


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
COLS = ["Ticker", "Country", "Asset", "1D", "5D", "1M", "3M", "YTD", "Z50", "MomZ", "Flag"]


def _fmt(v, pct: bool = True) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:+.2f}%" if pct else f"{v:+.2f}"


def _row(ticker: str, country: str, asset: str, ret: dict, brk: dict) -> list[str]:
    return [
        ticker, country, asset,
        *(_fmt(ret[h]) for h in ["1D", "5D", "1M", "3M", "YTD"]),
        _fmt(brk["Z50"], False), _fmt(brk["MomZ"], False), str(brk["Flag"]),
    ]


def render_table(rows: list[list[str]], fmt: str) -> str:
    if fmt == "markdown":
        head = "| " + " | ".join(COLS) + " |"
        sep = "|" + "|".join(["---"] * 3 + ["---:"] * 7 + [":---:"]) + "|"
        return "\n".join([head, sep] + ["| " + " | ".join(r) + " |" for r in rows])
    widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(COLS)]
    line = lambda r: "  ".join(c.ljust(widths[i]) if i < 3 else c.rjust(widths[i])
                               for i, c in enumerate(r))
    return "\n".join([line(COLS), "  ".join("-" * w for w in widths)] + [line(r) for r in rows])


def build_section(name: str, universe: dict, prices: pd.DataFrame, cfg: Config) -> str:
    avail = [t for t in universe if t in prices.columns and prices[t].dropna().size > MA_WINDOW + 63]
    missing = [t for t in universe if t not in avail]
    rows: list[list[str]] = []
    for t in avail:
        full = prices[t].dropna()
        win = full.iloc[-cfg.window:]
        rows.append(_row(t, universe[t][0], universe[t][1],
                         trailing_returns(full, win),
                         breakout_stats(full, cfg.z_threshold, cfg.window)))

    # Cluster baseline
    excl = COMPOSITE_EXCLUDE.get(name, set())
    comp_cols = [t for t in avail if t not in excl]
    comp = composite_index(prices[comp_cols], CUSTOM_WEIGHTS or None)
    comp_win = comp.iloc[-cfg.window:]
    base_label = "Custom-weight" if CUSTOM_WEIGHTS else "Equal-weight"
    rows.append(_row("COMPOSITE", f"{name} ({base_label}, n={len(comp_cols)})",
                     "Daily-rebalanced composite",
                     trailing_returns(comp, comp_win),
                     breakout_stats(comp, cfg.z_threshold, cfg.window)))

    title = f"{name.replace('_', ' ')} Heat Check"
    out = [f"## {title}" if cfg.fmt == "markdown" else f"=== {title} ===",
           render_table(rows, cfg.fmt)]
    if excl:
        out.append(f"Composite excludes: {', '.join(sorted(excl))} (constituent overlap)")
    if missing:
        out.append(f"Missing/insufficient data: {', '.join(missing)}")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Commentary (generated from the computed metrics; no external views)
# --------------------------------------------------------------------------- #
def collect_metrics(universe: dict, prices: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Numeric metrics per ticker (index = ticker) plus a COMPOSITE row."""
    rec = {}
    for t in universe:
        if t not in prices.columns or prices[t].dropna().size <= MA_WINDOW + 63:
            continue
        full = prices[t].dropna()
        rec[t] = {**trailing_returns(full, full.iloc[-cfg.window:]),
                  **breakout_stats(full, cfg.z_threshold, cfg.window)}
    return pd.DataFrame(rec).T


def commentary(name: str, universe: dict, prices: pd.DataFrame, cfg: Config) -> str:
    m = collect_metrics(universe, prices, cfg)
    if m.empty:
        return ""
    for c in ["1D", "5D", "1M", "3M", "YTD", "Z50", "MomZ"]:
        m[c] = m[c].astype(float)
    ctry = lambda t: f"{universe[t][0]} ({t})"
    n = len(m)
    lines = []

    # Breadth: share of markets with positive return per horizon.
    breadth = ", ".join(f"{h} {int((m[h] > 0).sum())}/{n}" for h in ["1D", "5D", "1M", "3M", "YTD"])
    lines.append(f"Breadth (markets with positive return): {breadth}.")

    # Leaders / laggards on 3M and YTD.
    for h in ["3M", "YTD"]:
        top, bot = m[h].idxmax(), m[h].idxmin()
        lines.append(f"{h} leader: {ctry(top)} {m.loc[top, h]:+.1f}%; "
                     f"laggard: {ctry(bot)} {m.loc[bot, h]:+.1f}%; "
                     f"spread {m.loc[top, h] - m.loc[bot, h]:.1f} pts; median {m[h].median():+.1f}%.")

    # Short-term vs medium-term disagreement (5D sign opposite to 3M sign).
    div = m[(np.sign(m["5D"]) != np.sign(m["3M"])) & (m["5D"].abs() > 0)]
    if not div.empty:
        lines.append("5D direction opposes 3M direction: "
                     + ", ".join(f"{universe[t][0]} (5D {r['5D']:+.1f}%, 3M {r['3M']:+.1f}%)"
                                 for t, r in div.iterrows()) + ".")

    # Statistical flags.
    hot = m[m["Flag"] != "-"]
    if hot.empty:
        lines.append(f"No market breaches the |z| >= {cfg.z_threshold} threshold on price or momentum.")
    else:
        lines.append(f"Breaching |z| >= {cfg.z_threshold}: "
                     + ", ".join(f"{universe[t][0]} [{r['Flag']}; Z50 {r['Z50']:+.2f}, MomZ {r['MomZ']:+.2f}]"
                                 for t, r in hot.iterrows()) + ".")
    near = m[(m["Flag"] == "-") & (m["Z50"].abs() >= 0.75 * cfg.z_threshold)]
    if not near.empty:
        lines.append("Within 25% of the price threshold: "
                     + ", ".join(f"{universe[t][0]} (Z50 {r['Z50']:+.2f})" for t, r in near.iterrows()) + ".")

    header = f"### {name.replace('_', ' ')} commentary" if cfg.fmt == "markdown" else f"--- {name} commentary ---"
    return header + "\n" + "\n".join(("- " if cfg.fmt == "markdown" else "* ") + l for l in lines)


# --------------------------------------------------------------------------- #
# Dashboards (one self-contained HTML file per cluster; no external assets)
# --------------------------------------------------------------------------- #
import html as _html
from dataclasses import replace as _replace

_CSS = """
:root{--bg:#fff;--fg:#1a1d21;--mut:#5b6470;--card:#f4f5f7;--line:#dde0e5;--up:46,160,90;--dn:215,60,60;--acc:#2b6cb0}
@media (prefers-color-scheme:dark){:root{--bg:#14171a;--fg:#e8eaed;--mut:#9aa3ad;--card:#1e2226;--line:#2d3339;--up:60,190,110;--dn:240,90,90;--acc:#63a4ee}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px}h1{font-size:22px;margin:0}h2{font-size:15px;margin:28px 0 10px}
.sub{color:var(--mut);margin:4px 0 20px}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}.tile b{display:block;font-size:20px;font-variant-numeric:tabular-nums}
.tile span{color:var(--mut);font-size:12px}.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;min-width:760px;font-variant-numeric:tabular-nums}th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600;font-size:12px;background:var(--card)}td:nth-child(-n+2),th:nth-child(-n+2){text-align:left}
tr.comp td{font-weight:700;border-top:2px solid var(--line)}.badge{display:inline-block;padding:1px 7px;border-radius:10px;border:1px solid var(--acc);color:var(--acc);font-size:12px;white-space:nowrap}
.zbar{position:relative;display:inline-block;width:90px;height:8px;background:var(--card);border:1px solid var(--line);border-radius:4px;vertical-align:middle;margin-left:6px}
.zbar i{position:absolute;top:0;bottom:0}.zbar u{position:absolute;top:-2px;bottom:-2px;width:1px;background:var(--mut)}
svg text{fill:var(--fg);font-size:12px}svg .m{fill:var(--mut)}ul{padding-left:18px;margin:0}li{margin:4px 0}.foot{color:var(--mut);font-size:12px;margin-top:24px}
@media(max-width:600px){main{padding:16px}}
"""


def _cell(v: float, scale: float) -> str:
    if pd.isna(v):
        return "<td>n/a</td>"
    a = 0.06 + 0.5 * min(abs(v) / scale, 1.0)
    rgb = "var(--up)" if v >= 0 else "var(--dn)"
    return f'<td style="background:rgba({rgb},{a:.2f})">{v:+.2f}%</td>'


def _zbar(z: float, thr: float, span: float = 3.5) -> str:
    if pd.isna(z):
        return "n/a"
    frac = max(-1.0, min(1.0, z / span))
    left, width = (50, frac * 50) if frac >= 0 else (50 + frac * 50, -frac * 50)
    col = "rgb(var(--up))" if z >= 0 else "rgb(var(--dn))"
    t = thr / span * 50
    return (f'{z:+.2f}<span class="zbar"><i style="left:{left:.1f}%;width:{width:.1f}%;background:{col}"></i>'
            f'<u style="left:{50 - t:.1f}%"></u><u style="left:50%"></u><u style="left:{50 + t:.1f}%"></u></span>')


def _ytd_chart(m: pd.DataFrame, universe: dict) -> str:
    d = m["YTD"].astype(float).sort_values(ascending=False)
    rowh, lw, w = 22, 215, 780
    mx = max(abs(d).max(), 1e-9)
    zero = lw + (w - lw) * (0.5 if d.min() < 0 < d.max() else (0.0 if d.min() >= 0 else 1.0))
    half = (w - lw) * (0.5 if d.min() < 0 < d.max() else 1.0)
    out = [f'<svg viewBox="0 0 {w} {rowh * len(d) + 4}" width="100%" role="img" aria-label="YTD return by market">']
    for i, (t, v) in enumerate(d.items()):
        y = i * rowh + 2
        bw = abs(v) / mx * (half - 50)
        x = zero if v >= 0 else zero - bw
        col = "rgb(var(--up))" if v >= 0 else "rgb(var(--dn))"
        lx = (x + bw + 4) if v >= 0 else (x - 4)
        anc = "start" if v >= 0 else "end"
        out.append(f'<text x="{lw - 8}" y="{y + 14}" text-anchor="end">{_html.escape(universe[t][0])} ({t})</text>'
                   f'<rect x="{x:.1f}" y="{y + 3}" width="{bw:.1f}" height="14" rx="2" fill="{col}"/>'
                   f'<text class="m" x="{lx:.1f}" y="{y + 14}" text-anchor="{anc}">{v:+.1f}%</text>')
    out.append(f'<line x1="{zero:.1f}" x2="{zero:.1f}" y1="0" y2="{rowh * len(d) + 4}" stroke="currentColor" opacity=".3"/></svg>')
    return "".join(out)


def render_dashboard(name: str, universe: dict, prices: pd.DataFrame, cfg: Config) -> str:
    """Self-contained HTML dashboard for one cluster (no network, no JS)."""
    m = collect_metrics(universe, prices, cfg)
    for c in ["1D", "5D", "1M", "3M", "YTD", "Z50", "MomZ"]:
        m[c] = m[c].astype(float)
    excl = COMPOSITE_EXCLUDE.get(name, set())
    comp_cols = [t for t in m.index if t not in excl]
    comp = composite_index(prices[comp_cols], CUSTOM_WEIGHTS or None)
    cm = {**trailing_returns(comp, comp.iloc[-cfg.window:]), **breakout_stats(comp, cfg.z_threshold, cfg.window)}
    scales = {h: max(float(m[h].abs().quantile(0.9)), 0.5) for h in ["1D", "5D", "1M", "3M", "YTD"]}
    title = name.replace("_", " ") + " Heat Check"
    asof = prices.index[-1].date()

    tiles = "".join(f'<div class="tile"><span>{h} composite</span><b style="color:rgb(var({"--up" if cm[h] >= 0 else "--dn"}))">{cm[h]:+.2f}%</b></div>'
                    for h in ["1D", "5D", "1M", "3M", "YTD"])
    tiles += f'<div class="tile"><span>Composite Z50 / flag</span><b>{cm["Z50"]:+.2f}</b><span>{_html.escape(str(cm["Flag"]))}</span></div>'

    def row(t, ctry, asset, r, cls=""):
        flag = "" if r["Flag"] == "-" else f'<span class="badge">{_html.escape(str(r["Flag"]))}</span>'
        cells = "".join(_cell(r[h], scales[h]) for h in ["1D", "5D", "1M", "3M", "YTD"])
        return (f'<tr class="{cls}"><td>{_html.escape(t)}</td><td>{_html.escape(ctry)}</td>{cells}'
                f'<td>{_zbar(r["Z50"], cfg.z_threshold)}</td><td>{r["MomZ"]:+.2f}</td><td>{flag}</td></tr>')

    body = "".join(row(t, universe[t][0], universe[t][1], r) for t, r in m.iterrows())
    body += row("COMPOSITE", f"{'Custom' if CUSTOM_WEIGHTS else 'Equal'}-weight, n={len(comp_cols)}", "", cm, "comp")

    text_cfg = _replace(cfg, fmt="text")
    bullets = commentary(name, universe, prices, text_cfg).splitlines()[1:]
    lis = "".join(f"<li>{_html.escape(b[2:])}</li>" for b in bullets)
    notes = f"Composite excludes: {', '.join(sorted(excl))} (constituent overlap). " if excl else ""

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{_CSS}</style></head><body><main>
<h1>{title}</h1><p class="sub">As of {asof} &middot; {cfg.window}-day window &middot; flag at |z| &ge; {cfg.z_threshold} &middot; USD adjusted closes</p>
<div class="tiles">{tiles}</div>
<h2>Returns and breakout statistics</h2><div class="wrap"><table><thead><tr><th>Ticker</th><th>Country</th><th>1D</th><th>5D</th><th>1M</th><th>3M</th><th>YTD</th><th>Z50 (price vs 50d MA)</th><th>MomZ</th><th>Flag</th></tr></thead><tbody>{body}</tbody></table></div>
<h2>YTD return by market</h2>{_ytd_chart(m, universe)}
<h2>Commentary (generated from the metrics)</h2><ul>{lis}</ul>
<p class="foot">{notes}Cell shading scales to each column's 90th-percentile absolute return. Z50 bar ticks mark 0 and &plusmn;{cfg.z_threshold}.</p>
</main></body></html>"""


def write_dashboards(prices: pd.DataFrame, cfg: Config, out_dir: str) -> list[str]:
    import os
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, universe in MARKET_GROUPS.items():
        path = os.path.join(out_dir, f"{name.split('_')[0].lower()}_dashboard.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(render_dashboard(name, universe, prices, cfg))
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="G7/G20 equity heat check")
    ap.add_argument("--window", type=int, default=252, help="trading days (default 252)")
    ap.add_argument("--z", type=float, default=2.0, help="z-score flag threshold")
    ap.add_argument("--no-commentary", action="store_true",
                    help="suppress the data-derived commentary blocks")
    ap.add_argument("--dashboard-dir", default=None,
                    help="write g7_dashboard.html and g20_dashboard.html to this directory")
    ap.add_argument("--format", choices=["markdown", "text"], default="markdown")
    a = ap.parse_args(argv)
    cfg = Config(a.window, a.z, a.format)

    prices = fetch_prices(sorted(G20_MARKETS), cfg.window)
    asof = prices.index[-1].date()
    print(f"# Global Market Heat Check -- as of {asof} | window={cfg.window}d | "
          f"flag |z|>={cfg.z_threshold} | MA={MA_WINDOW}d | prices=USD adj. close\n")
    for name, universe in MARKET_GROUPS.items():
        print(build_section(name, universe, prices, cfg), end="\n\n")
        if not a.no_commentary:
            c = commentary(name, universe, prices, cfg)
            if c:
                print(c, end="\n\n")
    if a.dashboard_dir:
        for pth in write_dashboards(prices, cfg, a.dashboard_dir):
            print(f"Dashboard written: {pth}")
    print("Legend: Z50 = (P-SMA50)/StDev50; MomZ = z of latest 5D return vs window; "
          "Flag PRICE+/- or MOM+/- when |z| >= threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
