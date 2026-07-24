# 这个文件用于实现MCP统一请求协议与工具调度流程。
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable
import json

from src.mcp.tools import ToolResult, calc_debt_ratio, calc_static_pe


ToolHandler = Callable[..., ToolResult]


class MCPDispatcher:
    # 这个类用于维护工具注册表并执行统一协议解析。
    def __init__(self) -> None:
        self._tools: dict[str, ToolHandler] = {
            "calc_pe": calc_static_pe,
            "calc_debt_ratio": calc_debt_ratio,
        }

    def list_tools(self) -> list[str]:
        # 这个方法用于返回当前可用工具列表。
        return sorted(self._tools.keys())

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        # 这个方法用于执行一次MCP请求并返回统一响应。
        try:
            tool_name = request.get("tool_name", "")
            params = request.get("params", {})
            request_id = request.get("request_id")

            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError("字段tool_name不能为空字符串。")
            if not isinstance(params, dict):
                raise ValueError("字段params必须是JSON对象。")

            handler = self._tools.get(tool_name)
            if handler is None:
                raise KeyError(f"未知工具: {tool_name}，可用工具: {', '.join(self.list_tools())}")

            result = handler(**params)
            return {
                "ok": True,
                "tool_name": tool_name,
                "request_id": request_id,
                "result": asdict(result),
                "error": None,
            }
        except TypeError as exc:
            return self._error_response(request, "INVALID_PARAMS", f"参数不匹配: {exc}")
        except (ValueError, KeyError) as exc:
            return self._error_response(request, "BAD_REQUEST", str(exc))
        except Exception as exc:  # noqa: BLE001
            return self._error_response(request, "INTERNAL_ERROR", f"工具执行失败: {exc}")

    def handle_json(self, payload: str) -> dict[str, Any]:
        # 这个方法用于接收JSON字符串并完成解析与调度。
        try:
            request = json.loads(payload)
        except json.JSONDecodeError as exc:
            return {
                "ok": False,
                "tool_name": None,
                "request_id": None,
                "result": None,
                "error": {
                    "code": "INVALID_JSON",
                    "message": f"请求体不是合法JSON: {exc.msg}",
                },
            }

        if not isinstance(request, dict):
            return {
                "ok": False,
                "tool_name": None,
                "request_id": None,
                "result": None,
                "error": {
                    "code": "INVALID_JSON",
                    "message": "请求体顶层必须是JSON对象。",
                },
            }
        return self.handle(request)

    @staticmethod
    def _error_response(request: dict[str, Any], code: str, message: str) -> dict[str, Any]:
        # 这个方法用于统一错误响应结构，便于上层稳定处理。
        return {
            "ok": False,
            "tool_name": request.get("tool_name"),
            "request_id": request.get("request_id"),
            "result": None,
            "error": {
                "code": code,
                "message": message,
            },
        }
