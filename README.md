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
