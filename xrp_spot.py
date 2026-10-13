#!/usr/bin/env python3
"""Fetch the XRP/USD spot price from public APIs (no key needed).

Tries Coinbase, then Kraken. Prints the price; with --nav and
--xrp-per-share it also shows what XRPZ NAV that price implies.

    python3 xrp_spot.py
    python3 xrp_spot.py --nav 15.03 --xrp-per-share 10.85
"""
import argparse
import json
import sys
import urllib.request

TIMEOUT = 10


def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "kgfo/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.load(resp)


def coinbase():
    data = _get_json("https://api.coinbase.com/v2/prices/XRP-USD/spot")
    return float(data["data"]["amount"])


def kraken():
    data = _get_json("https://api.kraken.com/0/public/Ticker?pair=XRPUSD")
    if data.get("error"):
        raise RuntimeError(data["error"])
    (ticker,) = data["result"].values()
    return float(ticker["c"][0])  # last trade price


SOURCES = [("Coinbase", coinbase), ("Kraken", kraken)]


def fetch_spot():
    """Return (price, source name). Raises RuntimeError if every source fails."""
    errors = []
    for name, fn in SOURCES:
        try:
            return fn(), name
        except Exception as e:  # network, HTTP, or unexpected payload
            errors.append(f"{name}: {e}")
    raise RuntimeError("all sources failed: " + "; ".join(errors))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nav", type=float, help="XRPZ NAV per share, to compare with spot")
    ap.add_argument("--xrp-per-share", type=float, default=10.85)
    args = ap.parse_args()

    try:
        price, source = fetch_spot()
    except RuntimeError as e:
        sys.exit(str(e))

    print(f"XRP/USD  ${price:,.4f}  ({source})")
    implied_nav = price * args.xrp_per_share
    print(f"implied XRPZ NAV  ${implied_nav:,.2f}  ({args.xrp_per_share} XRP/share)")
    if args.nav:
        gap = (args.nav / implied_nav - 1) * 100
        print(f"given NAV ${args.nav:,.2f} vs implied: {gap:+.2f}%")


if __name__ == "__main__":
    main()
