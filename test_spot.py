"""Offline check of xrp_spot.py: fake HTTP responses, no network."""
import io
import json
import sys
import urllib.request

import xrp_spot

CB = {"data": {"base": "XRP", "currency": "USD", "amount": "1.3850"}}
KR = {"error": [], "result": {"XXRPZUSD": {"c": ["1.3900", "100"]}}}


def fake_urlopen(payload, fail_for=None):
    def _open(req, timeout=None):
        if fail_for and fail_for in req.full_url:
            raise OSError("simulated outage")
        body = KR if "kraken" in req.full_url else CB
        return io.BytesIO(json.dumps(payload or body).encode())
    return _open


# 1. Coinbase works
urllib.request.urlopen = fake_urlopen(None)
assert xrp_spot.fetch_spot() == (1.385, "Coinbase")

# 2. Coinbase down -> Kraken fallback
urllib.request.urlopen = fake_urlopen(None, fail_for="coinbase")
assert xrp_spot.fetch_spot() == (1.39, "Kraken")

# 3. Both down -> RuntimeError
def all_down(req, timeout=None):
    raise OSError("simulated outage")
urllib.request.urlopen = all_down
try:
    xrp_spot.fetch_spot()
except RuntimeError as e:
    assert "all sources failed" in str(e)
else:
    sys.exit("expected RuntimeError")

print("xrp_spot: 3/3 checks passed")
