import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


def _unpatched_stub(*args, **kwargs):  # pragma: no cover - defensive stub
    raise RuntimeError("stub should be patched in tests")


if "baostock" not in sys.modules:
    baostock_stub = types.ModuleType("baostock")
    baostock_stub.login = _unpatched_stub
    baostock_stub.logout = _unpatched_stub
    baostock_stub.query_history_k_data_plus = _unpatched_stub
    baostock_stub.query_balance_data = _unpatched_stub
    baostock_stub.query_cash_flow_data = _unpatched_stub
    baostock_stub.query_profit_data = _unpatched_stub
    baostock_stub.query_growth_data = _unpatched_stub
    sys.modules["baostock"] = baostock_stub

if "akshare" not in sys.modules:
    akshare_stub = types.ModuleType("akshare")
    akshare_stub.stock_zh_a_daily = _unpatched_stub
    akshare_stub.stock_financial_abstract = _unpatched_stub
    akshare_stub.stock_financial_report_sina = _unpatched_stub
    sys.modules["akshare"] = akshare_stub

from tradingagents.dataflows import interface
from tradingagents.dataflows import config as data_config


class _DummyFrame:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.empty = False

    def to_csv(self, *args, **kwargs):  # pragma: no cover - simple proxy
        return self.payload


class _DummyQuery:
    def __init__(self, fields, rows):
        self.fields = fields
        self._rows = list(rows)
        self.error_code = "0"
        self.error_msg = ""
        self._index = 0

    def next(self):
        if self._index < len(self._rows):
            self._current = self._rows[self._index]
            self._index += 1
            return True
        return False

    def get_row_data(self):
        return self._current


class DataflowVendorRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        importlib.reload(data_config)
        importlib.reload(interface)
        self.addCleanup(lambda: importlib.reload(data_config))
        self.addCleanup(lambda: importlib.reload(interface))

    @patch("tradingagents.dataflows.akshare.ak.stock_zh_a_daily")
    def test_route_to_vendor_akshare_stock(self, mock_hist):
        mock_hist.return_value = _DummyFrame("date,open\n2024-01-02,10")

        data_config.set_config({"tool_vendors": {"get_stock_data": "akshare"}})

        result = interface.route_to_vendor(
            "get_stock_data", "600000.SH", "2024-01-01", "2024-01-05"
        )

        self.assertIn("AKShare daily data", result)
        mock_hist.assert_called_once_with(
            symbol="sh600000", start_date="20240101", end_date="20240105", adjust=""
        )

    @patch("tradingagents.dataflows.akshare.ak.stock_financial_abstract")
    def test_route_to_vendor_akshare_fundamentals(self, mock_financial_abstract):
        mock_financial_abstract.return_value = _DummyFrame("field,value\nroe,12.5")

        data_config.set_config({"tool_vendors": {"get_fundamentals": "akshare"}})

        result = interface.route_to_vendor("get_fundamentals", "600000.SH")

        self.assertIn("AKShare financial abstract", result)
        mock_financial_abstract.assert_called_once_with(symbol="600000")

    @patch("tradingagents.dataflows.akshare.ak.stock_zh_a_daily")
    def test_route_to_vendor_akshare_stock_alternate_suffix(self, mock_hist):
        mock_hist.return_value = _DummyFrame("date,open\n2024-01-02,10")

        data_config.set_config({"tool_vendors": {"get_stock_data": "akshare"}})

        interface.route_to_vendor("get_stock_data", "600000.SS", "2024-01-01", "2024-01-05")

        mock_hist.assert_called_once_with(
            symbol="sh600000", start_date="20240101", end_date="20240105", adjust=""
        )

    @patch("tradingagents.dataflows.akshare.ak.stock_zh_a_daily")
    def test_route_to_vendor_akshare_stock_xshg_suffix(self, mock_hist):
        mock_hist.return_value = _DummyFrame("date,open\n2024-01-02,10")

        data_config.set_config({"tool_vendors": {"get_stock_data": "akshare"}})

        interface.route_to_vendor("get_stock_data", "688001.XSHG", "2024-01-01", "2024-01-05")

        mock_hist.assert_called_once_with(
            symbol="sh688001", start_date="20240101", end_date="20240105", adjust=""
        )

    @patch("tradingagents.dataflows.akshare.ak.stock_zh_a_daily")
    def test_route_to_vendor_akshare_stock_xshe_suffix(self, mock_hist):
        mock_hist.return_value = _DummyFrame("date,open\n2024-01-02,10")

        data_config.set_config({"tool_vendors": {"get_stock_data": "akshare"}})

        interface.route_to_vendor("get_stock_data", "000001.XSHE", "2024-01-01", "2024-01-05")

        mock_hist.assert_called_once_with(
            symbol="sz000001", start_date="20240101", end_date="20240105", adjust=""
        )

    @patch("tradingagents.dataflows.akshare.ak.stock_zh_a_daily")
    def test_route_to_vendor_akshare_stock_beijing_suffix(self, mock_hist):
        mock_hist.return_value = _DummyFrame("date,open\n2024-01-02,10")

        data_config.set_config({"tool_vendors": {"get_stock_data": "akshare"}})

        interface.route_to_vendor(
            "get_stock_data",
            "430047.BJ",
            "2024-01-01",
            "2024-01-05",
            adjust="hfq",
        )

        mock_hist.assert_called_once_with(
            symbol="bj430047", start_date="20240101", end_date="20240105", adjust="hfq"
        )

    @patch("tradingagents.dataflows.akshare.ak.stock_financial_report_sina")
    def test_route_to_vendor_akshare_balance_sheet(self, mock_report):
        mock_report.return_value = _DummyFrame("field,value\nassets,100")

        data_config.set_config({"tool_vendors": {"get_balance_sheet": "akshare"}})

        result = interface.route_to_vendor("get_balance_sheet", "600519.SH")

        self.assertIn("AKShare balance sheet", result)
        mock_report.assert_called_once_with(stock="600519", symbol="资产负债表")

    @patch("tradingagents.dataflows.akshare.ak.stock_financial_report_sina")
    def test_route_to_vendor_akshare_balance_sheet_beijing(self, mock_report):
        mock_report.return_value = _DummyFrame("field,value\nassets,100")

        data_config.set_config({"tool_vendors": {"get_balance_sheet": "akshare"}})

        result = interface.route_to_vendor("get_balance_sheet", "430047.BJ")

        self.assertIn("AKShare balance sheet", result)
        mock_report.assert_called_once_with(stock="430047", symbol="资产负债表")

    @patch("tradingagents.dataflows.akshare.ak.stock_financial_report_sina")
    def test_route_to_vendor_akshare_reports_empty_frame(self, mock_report):
        class _EmptyFrame:
            empty = True

        mock_report.return_value = _EmptyFrame()

        data_config.set_config({"tool_vendors": {"get_balance_sheet": "akshare"}})

        result = interface.route_to_vendor("get_balance_sheet", "430047.BJ")

        self.assertIn("no akshare balance sheet available", result.lower())

    @patch("tradingagents.dataflows.akshare.ak.stock_financial_report_sina")
    def test_route_to_vendor_akshare_cashflow(self, mock_report):
        mock_report.return_value = _DummyFrame("field,value\ncash,200")

        data_config.set_config({"tool_vendors": {"get_cashflow": "akshare"}})

        result = interface.route_to_vendor("get_cashflow", "300750.SZ")

        self.assertIn("AKShare cashflow statement", result)
        mock_report.assert_called_once_with(stock="300750", symbol="现金流量表")

    @patch("tradingagents.dataflows.akshare.ak.stock_financial_report_sina")
    def test_route_to_vendor_akshare_income_statement(self, mock_report):
        mock_report.return_value = _DummyFrame("field,value\nnet_profit,50")

        data_config.set_config({"tool_vendors": {"get_income_statement": "akshare"}})

        result = interface.route_to_vendor("get_income_statement", "000001.SZ")

        self.assertIn("AKShare income statement", result)
        mock_report.assert_called_once_with(stock="000001", symbol="利润表")

    @patch("tradingagents.dataflows.baostock._load_baostock")
    def test_route_to_vendor_baostock_stock(self, mock_load):
        login_mock = Mock(return_value=SimpleNamespace(error_code="0", error_msg=""))
        logout_mock = Mock()
        query_mock = Mock(return_value=_DummyQuery(["date", "open"], [["2024-01-02", "11"]]))
        mock_load.return_value = SimpleNamespace(
            login=login_mock,
            logout=logout_mock,
            query_history_k_data_plus=query_mock,
        )

        data_config.set_config({"tool_vendors": {"get_stock_data": "baostock"}})

        result = interface.route_to_vendor(
            "get_stock_data", "000001.SZ", "2024-01-01", "2024-01-03"
        )

        self.assertIn("BaoStock d price data", result)
        login_mock.assert_called()
        logout_mock.assert_called()

    @patch("tradingagents.dataflows.baostock._load_baostock")
    def test_route_to_vendor_baostock_balance_sheet(self, mock_load):
        login_mock = Mock(return_value=SimpleNamespace(error_code="0", error_msg=""))
        logout_mock = Mock()
        balance_mock = Mock(
            return_value=_DummyQuery(["field", "value"], [["assets", "100"]])
        )
        mock_load.return_value = SimpleNamespace(
            login=login_mock,
            logout=logout_mock,
            query_balance_data=balance_mock,
        )

        data_config.set_config({"tool_vendors": {"get_balance_sheet": "baostock"}})

        result = interface.route_to_vendor("get_balance_sheet", "600519.SH")

        self.assertIn("BaoStock balance sheet", result)
        balance_mock.assert_called()

    @patch("tradingagents.dataflows.baostock._load_baostock")
    def test_route_to_vendor_baostock_fundamentals(self, mock_load):
        login_mock = Mock(return_value=SimpleNamespace(error_code="0", error_msg=""))
        logout_mock = Mock()
        growth_mock = Mock(
            return_value=_DummyQuery(["field", "value"], [["roe", "15"]])
        )
        mock_load.return_value = SimpleNamespace(
            login=login_mock,
            logout=logout_mock,
            query_growth_data=growth_mock,
        )

        data_config.set_config({"tool_vendors": {"get_fundamentals": "baostock"}})

        result = interface.route_to_vendor("get_fundamentals", "600000.SH")

        self.assertIn("BaoStock growth indicators", result)
        growth_mock.assert_called()

    @patch("tradingagents.dataflows.baostock._load_baostock")
    def test_route_to_vendor_baostock_cashflow(self, mock_load):
        login_mock = Mock(return_value=SimpleNamespace(error_code="0", error_msg=""))
        logout_mock = Mock()
        cashflow_mock = Mock(
            return_value=_DummyQuery(["field", "value"], [["net", "20"]])
        )
        mock_load.return_value = SimpleNamespace(
            login=login_mock,
            logout=logout_mock,
            query_cash_flow_data=cashflow_mock,
        )

        data_config.set_config({"tool_vendors": {"get_cashflow": "baostock"}})

        result = interface.route_to_vendor("get_cashflow", "600519.SH")

        self.assertIn("BaoStock cashflow statement", result)
        cashflow_mock.assert_called()

    @patch("tradingagents.dataflows.baostock._load_baostock")
    def test_route_to_vendor_baostock_income_statement(self, mock_load):
        login_mock = Mock(return_value=SimpleNamespace(error_code="0", error_msg=""))
        logout_mock = Mock()
        profit_mock = Mock(
            return_value=_DummyQuery(["field", "value"], [["revenue", "30"]])
        )
        mock_load.return_value = SimpleNamespace(
            login=login_mock,
            logout=logout_mock,
            query_profit_data=profit_mock,
        )

        data_config.set_config({"tool_vendors": {"get_income_statement": "baostock"}})

        result = interface.route_to_vendor("get_income_statement", "000001.SZ")

        self.assertIn("BaoStock income statement", result)
        profit_mock.assert_called()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
