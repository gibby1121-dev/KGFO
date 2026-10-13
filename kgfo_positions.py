#!/usr/bin/env python3
"""Read-only Schwab positions report with XRPZ look-through.

Prints every position, then for XRPZ: shares, cost basis, market value,
XRP-equivalent exposure, and premium/discount to NAV (if --nav is given).

Setup (on your own machine, never commit keys or the token file):
    pip install schwab-py
    export SCHWAB_APP_KEY=...  SCHWAB_APP_SECRET=...
    python3 kgfo_positions.py --nav 15.03

First run opens a browser login and writes schwab_token.json.
The refresh token expires after 7 days; delete the file to log in again.
This script never places orders.
"""
import argparse
import os
import sys

from schwab import auth

CALLBACK_URL = "https://127.0.0.1:8182"
TOKEN_PATH = "schwab_token.json"
SYMBOL = "XRPZ"


def money(x):
    """Dollar string with the sign in front: -$2,800.00, not $-2,800.00."""
    return f"{'-' if x < 0 else ''}${abs(x):,.2f}"


def get_client():
    try:
        key = os.environ["SCHWAB_APP_KEY"]
        secret = os.environ["SCHWAB_APP_SECRET"]
    except KeyError as e:
        sys.exit(f"Missing environment variable: {e.args[0]}")
    return auth.easy_client(key, secret, CALLBACK_URL, TOKEN_PATH)


def positions(client):
    resp = client.get_accounts(fields=[client.Account.Fields.POSITIONS])
    resp.raise_for_status()
    for acct in resp.json():
        sec = acct["securitiesAccount"]
        # Account numbers are masked to the last 4 digits in output.
        label = "..." + str(sec.get("accountNumber", ""))[-4:]
        for p in sec.get("positions", []):
            yield label, p


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nav", type=float, help="XRPZ NAV per share (from Franklin's site)")
    ap.add_argument("--xrp-per-share", type=float, default=10.85,
                    help="XRP backing each share (default 10.85; decays slowly with fees)")
    args = ap.parse_args()

    client = get_client()

    print(f"{'acct':<8}{'symbol':<10}{'qty':>12}{'avg cost':>12}{'mkt value':>14}")
    xrpz_qty = xrpz_cost = xrpz_value = 0.0
    for label, p in positions(client):
        sym = p["instrument"]["symbol"]
        qty = p.get("longQuantity", 0.0) - p.get("shortQuantity", 0.0)
        avg = p.get("averagePrice", 0.0)
        val = p.get("marketValue", 0.0)
        print(f"{label:<8}{sym:<10}{qty:>12,.2f}{avg:>12,.2f}{val:>14,.2f}")
        if sym == SYMBOL:
            xrpz_qty += qty
            xrpz_cost += qty * avg
            xrpz_value += val

    if not xrpz_qty:
        print(f"\nNo {SYMBOL} position found.")
        return

    price = xrpz_value / xrpz_qty
    print(f"\n{SYMBOL} look-through")
    print(f"  shares              {xrpz_qty:,.2f}")
    print(f"  cost basis          ${xrpz_cost:,.2f}")
    print(f"  market value        ${xrpz_value:,.2f}  (${price:,.2f}/sh)")
    print(f"  unrealized P/L      {money(xrpz_value - xrpz_cost)}")
    print(f"  XRP-equivalent      {xrpz_qty * args.xrp_per_share:,.1f} XRP"
          f"  ({args.xrp_per_share} XRP/share)")
    if args.nav:
        prem = (price / args.nav - 1) * 100
        print(f"  vs NAV ${args.nav:,.2f}     {prem:+.2f}% premium/discount")
        print(f"  implied XRP price   ${args.nav / args.xrp_per_share:,.4f}")


if __name__ == "__main__":
    main()
