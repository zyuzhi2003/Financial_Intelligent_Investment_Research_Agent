# 这个脚本用于独立验证阶段三MCP工具协议与计算结果。
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

# 这个代码块用于把项目根目录加入导入路径，保证脚本可独立执行。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mcp import MCPDispatcher
from src.mcp.nl_router import build_request_from_question


def _assert(condition: bool, message: str) -> None:
    # 这个函数用于输出断言失败信息并中断测试。
    if not condition:
        raise AssertionError(message)


def main() -> int:
    # 这个函数用于执行全部MCP工具测试用例。
    dispatcher = MCPDispatcher()

    pe_response = dispatcher.handle(
        {
            "tool_name": "calc_pe",
            "params": {"price": 8.25, "eps": 0.68},
            "request_id": "case-pe-001",
        }
    )
    _assert(pe_response["ok"] is True, "PE合法输入应成功")
    pe_value = pe_response["result"]["value"]
    _assert(math.isclose(pe_value, 12.1323529412, rel_tol=1e-9), "PE结果不符合预期")

    debt_response = dispatcher.handle(
        {
            "tool_name": "calc_debt_ratio",
            "params": {"total_asset": 217867, "total_liability": 199242},
            "request_id": "case-debt-001",
        }
    )
    _assert(debt_response["ok"] is True, "资产负债率合法输入应成功")
    debt_value = debt_response["result"]["value"]
    _assert(math.isclose(debt_value, 0.9145120647, rel_tol=1e-9), "资产负债率结果不符合预期")

    invalid_zero_divisor = dispatcher.handle(
        {
            "tool_name": "calc_pe",
            "params": {"price": 8.25, "eps": 0},
            "request_id": "case-pe-invalid-001",
        }
    )
    _assert(invalid_zero_divisor["ok"] is False, "EPS为0应返回失败")

    invalid_negative = dispatcher.handle(
        {
            "tool_name": "calc_debt_ratio",
            "params": {"total_asset": 217867, "total_liability": -1},
            "request_id": "case-debt-invalid-001",
        }
    )
    _assert(invalid_negative["ok"] is False, "负数输入应返回失败")

    invalid_json = dispatcher.handle_json("not-json")
    _assert(invalid_json["ok"] is False, "非法JSON应返回失败")

    valuation_request = build_request_from_question(
        "请结合研报判断中信银行当前估值高低，已知股价8.25元、EPS0.68"
    )
    _assert(valuation_request["tool_name"] == "calc_pe", "估值语义应映射到PE计算")
    _assert(math.isclose(valuation_request["params"]["price"], 8.25, rel_tol=1e-9), "股价提取异常")
    _assert(math.isclose(valuation_request["params"]["eps"], 0.68, rel_tol=1e-9), "EPS提取异常")

    print("MCP测试通过：")
    print(json.dumps(pe_response, ensure_ascii=False, indent=2))
    print(json.dumps(debt_response, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"MCP测试失败: {exc}", file=sys.stderr)
        raise SystemExit(1)
