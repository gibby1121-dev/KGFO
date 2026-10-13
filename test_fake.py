"""Offline check of kgfo_positions.py using a fake Schwab client (no network)."""
import os
import runpy
import sys
import types

ACCOUNTS = [{
    "securitiesAccount": {
        "accountNumber": "12345678",
        "positions": [
            {"instrument": {"symbol": "XRPZ"}, "longQuantity": 1000.0,
             "averagePrice": 18.00, "marketValue": 15200.00},
            {"instrument": {"symbol": "AAPL"}, "longQuantity": 10.0,
             "averagePrice": 150.0, "marketValue": 2500.0},
        ],
    }
}]


class FakeResp:
    def raise_for_status(self):
        pass

    def json(self):
        return ACCOUNTS


class FakeClient:
    class Account:
        class Fields:
            POSITIONS = "positions"

    def get_accounts(self, fields=None):
        return FakeResp()


fake_auth = types.SimpleNamespace(easy_client=lambda *a, **k: FakeClient())
fake_schwab = types.ModuleType("schwab")
fake_schwab.auth = fake_auth
sys.modules["schwab"] = fake_schwab

os.environ["SCHWAB_APP_KEY"] = "x"
os.environ["SCHWAB_APP_SECRET"] = "y"
sys.argv = ["kgfo_positions.py", "--nav", "15.03"]
runpy.run_path(os.path.join(os.path.dirname(__file__), "kgfo_positions.py"),
               run_name="__main__")
