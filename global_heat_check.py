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
    close = close.dropna(how="all").ffill()
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="G7/G20 equity heat check")
    ap.add_argument("--window", type=int, default=252, help="trading days (default 252)")
    ap.add_argument("--z", type=float, default=2.0, help="z-score flag threshold")
    ap.add_argument("--no-commentary", action="store_true",
                    help="suppress the data-derived commentary blocks")
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
    print("Legend: Z50 = (P-SMA50)/StDev50; MomZ = z of latest 5D return vs window; "
          "Flag PRICE+/- or MOM+/- when |z| >= threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
