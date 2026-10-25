"""Phase 7.6 — DevAgent as an MCP server.

Exposes all registered DevAgent tools as MCP tools so that any MCP client
(Claude Desktop, Cursor, Continue, etc.) can invoke the full DevAgent
tool-set including file editing, shell execution, git operations, and
CodePrism graph queries.

Usage
-----
stdio (default, for Claude Desktop / Cursor):
    devagent mcp

SSE (for web clients):
    devagent mcp --transport sse --port 7332

Claude Desktop config (claude_desktop_config.json):
    {
      "mcpServers": {
        "devagent": {
          "command": "devagent",
          "args": ["mcp"],
          "env": {}
        }
      }
    }
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from devagent.core.llm import ToolDef


def _json_schema_to_fastmcp_params(tool_def: ToolDef) -> dict[str, Any]:
    """Extract a flat {param: annotation_str} dict from a JSON-schema parameters block.

    FastMCP accepts Python type-annotated functions; since we're building
    tools dynamically we use a workaround: pass the raw JSON schema via
    the FastMCP `tool()` decorator's `parameters` kwarg when available,
    otherwise we register a generic **kwargs handler and document the
    schema in the description.
    """
    return tool_def.parameters or {}


def _make_handler(registry, tool_name: str):
    """Return a closure that calls registry.call(tool_name, args)."""

    def handler(**kwargs: Any) -> str:
        return registry.call(tool_name, dict(kwargs))

    handler.__name__ = tool_name
    return handler


def serve_mcp(
    project_root: str,
    config: Any,  # DevAgentConfig
    transport: str = "stdio",
    port: int = 7332,
) -> None:
    """Start DevAgent as an MCP server.

    Parameters
    ----------
    project_root:
        Absolute path to the project directory the agent operates on.
    config:
        DevAgentConfig instance (used to wire CodePrism / GitHub etc.).
    transport:
        "stdio" (default) or "sse".
    port:
        TCP port for SSE transport (ignored for stdio).
    """
    from devagent.tools.registry import build_registry

    # ------------------------------------------------------------------
    # Build the tool registry (same as a normal agent session)
    # ------------------------------------------------------------------
    codeprism_client = None
    github_token: str | None = None

    try:
        from devagent.codeprism.client import CodePrismClient
        codeprism_client = CodePrismClient(project_root)
    except Exception:
        pass

    try:
        github_token = config.github.token or None
    except AttributeError:
        pass

    registry = build_registry(
        project_root=project_root,
        codeprism_client=codeprism_client,
        github_token=github_token,
    )

    # ------------------------------------------------------------------
    # Create FastMCP server
    # ------------------------------------------------------------------
    server = FastMCP(
        "DevAgent",
        instructions=(
            "DevAgent tool-set for the project at: " + project_root + ".\n"
            "Use these tools to read/write files, run shell commands, "
            "query the CodePrism code graph, and manage git operations."
        ),
    )

    # ------------------------------------------------------------------
    # Dynamically register every tool from the registry
    # ------------------------------------------------------------------
    tool_defs: list[ToolDef] = registry.get_definitions()

    for td in tool_defs:
        _register_tool(server, registry, td)

    # ------------------------------------------------------------------
    # Expose CodePrism graph as MCP resources (if available)
    # ------------------------------------------------------------------
    if codeprism_client is not None:
        _register_codeprism_resources(server, codeprism_client, project_root)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    if transport == "sse":
        server.run(transport="sse", port=port)
    else:
        server.run(transport="stdio")


def _register_tool(server: FastMCP, registry: Any, td: ToolDef) -> None:
    """Register one ToolDef as an MCP tool on the FastMCP server.

    FastMCP needs a real Python callable.  Since our tools accept arbitrary
    JSON args we use a single `args_json` string parameter and decode it
    inside the handler — this keeps things simple while preserving full
    flexibility.  The JSON schema from the ToolDef is embedded in the
    tool description so the LLM client still sees the proper schema.
    """
    schema_str = json.dumps(td.parameters, indent=2) if td.parameters else "{}"
    description = (
        f"{td.description}\n\n"
        f"**Parameters (JSON schema)**:\n```json\n{schema_str}\n```\n\n"
        "Pass all parameters as a JSON object string in `args_json`."
    )

    tool_name = td.name

    # Capture tool_name in closure
    def _handler(args_json: str = "{}") -> str:
        """args_json: JSON object of tool arguments."""
        try:
            args = json.loads(args_json)
        except json.JSONDecodeError as exc:
            return f"[tool_error] Invalid JSON in args_json: {exc}"
        return registry.call(tool_name, args)

    _handler.__name__ = tool_name
    _handler.__doc__ = description

    server.tool(name=tool_name, description=description)(_handler)


def _register_codeprism_resources(
    server: FastMCP,
    codeprism_client: Any,
    project_root: str,
) -> None:
    """Expose key CodePrism graph data as MCP resources."""

    @server.resource("codeprism://graph/summary")
    def graph_summary() -> str:
        """High-level summary of the CodePrism graph for this project."""
        try:
            summary = codeprism_client.get_summary()
            return json.dumps(summary, indent=2)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @server.resource("codeprism://graph/symbols")
    def graph_symbols() -> str:
        """All symbols (functions, classes, etc.) indexed by CodePrism."""
        try:
            symbols = codeprism_client.list_symbols()
            return json.dumps(symbols, indent=2)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @server.resource("codeprism://graph/dependencies")
    def graph_dependencies() -> str:
        """Module dependency graph for this project."""
        try:
            deps = codeprism_client.get_dependencies()
            return json.dumps(deps, indent=2)
        except Exception as exc:
            return json.dumps({"error": str(exc)})
