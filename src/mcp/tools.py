# 这个文件用于实现MCP层可插拔的财务计算工具函数。
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolResult:
    # 这个字段用于返回核心数值结果。
    value: float
    # 这个字段用于返回展示给用户的计算公式文本。
    formula: str
    # 这个字段用于返回简短的业务解读。
    interpretation: str


def _to_positive_number(name: str, value: object) -> float:
    # 这个函数用于统一校验参数是否为合法正数。
    if not isinstance(value, (int, float)):
        raise ValueError(f"参数{name}必须是数值类型。")

    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"参数{name}必须是有限数值。")
    if number <= 0:
        raise ValueError(f"参数{name}必须大于0。")
    return number


def _to_non_negative_number(name: str, value: object) -> float:
    # 这个函数用于统一校验参数是否为合法非负数。
    if not isinstance(value, (int, float)):
        raise ValueError(f"参数{name}必须是数值类型。")

    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"参数{name}必须是有限数值。")
    if number < 0:
        raise ValueError(f"参数{name}不能为负数。")
    return number


def calc_static_pe(*, price: object, eps: object) -> ToolResult:
    # 这个函数用于计算静态PE并返回可解释结果。
    price_number = _to_positive_number("price", price)
    eps_number = _to_positive_number("eps", eps)

    pe = price_number / eps_number
    return ToolResult(
        value=pe,
        formula=f"静态PE = 股价 ÷ EPS = {price_number:.4g} ÷ {eps_number:.4g} = {pe:.4f}",
        interpretation="PE数值越高通常代表估值预期更高，需结合行业均值综合判断。",
    )


def calc_debt_ratio(*, total_asset: object, total_liability: object) -> ToolResult:
    # 这个函数用于计算资产负债率并返回可解释结果。
    asset_number = _to_positive_number("total_asset", total_asset)
    liability_number = _to_non_negative_number("total_liability", total_liability)

    ratio = liability_number / asset_number
    percent = ratio * 100
    return ToolResult(
        value=ratio,
        formula=(
            "资产负债率 = 总负债 ÷ 总资产 = "
            f"{liability_number:.4g} ÷ {asset_number:.4g} = {percent:.4f}%"
        ),
        interpretation="资产负债率反映杠杆水平，需与同业区间和负债结构一起分析。",
    )
