# KGFO

Read-only Schwab positions report with XRPZ (Franklin XRP ETF) look-through:
shares, cost basis, market value, XRP-equivalent exposure, and premium/discount to NAV.

```bash
pip install -r requirements.txt
export SCHWAB_APP_KEY=...  SCHWAB_APP_SECRET=...
python3 kgfo_positions.py --nav 15.03
```

- First run opens a browser login and writes `schwab_token.json` (git-ignored; refresh token lasts 7 days).
- Never places orders. Never commit keys or the token file.
- `python3 test_fake.py` runs the report against fake data, offline.

## Getting the NAV

Schwab does not provide XRPZ's NAV, so you pass it in with `--nav`.

- **Where:** the [Franklin XRP ETF page](https://www.franklintempleton.com/investments/options/exchange-traded-funds/products/47318/SINGLCLASS/franklin-xrp-etf/XRPZ) shows NAV per share and its as-of date.
- **Use NAV, not market price.** The page lists both. The script compares your Schwab price to the NAV you give it.
- **NAV is struck after the close**, so during the day the latest figure is from the prior trading day. Check the as-of date. Other sites (Yahoo, TradingView) can differ or lag.
- **Cross-check:** NAV should be about `XRP per share x XRP spot price`. At 10.85 XRP per share, a $15.03 NAV implies XRP near $1.385. The script prints this implied price, so compare it with the live XRP price. A large gap means the NAV is stale or mistyped.
- **XRP per share:** `--xrp-per-share` defaults to 10.85. The exact figure is XRP held divided by shares outstanding, from the fund's latest 10-Q or 10-K on SEC EDGAR (Franklin XRP Trust). It drifts down slightly as fees are paid in XRP.

```bash
python3 kgfo_positions.py --nav 15.03 --xrp-per-share 10.85
```
