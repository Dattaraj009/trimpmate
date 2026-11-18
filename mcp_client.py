import asyncio
import hashlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import certifi
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient


# =========================================================
# Environment setup
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Support both environment-variable names.
AVIATION_STACK_API_KEY = (
    os.getenv("AVIATION_STACK_API_KEY")
    or os.getenv("AVIATIONSTACK_API_KEY")
)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
UPSTASH_REDIS_REST_URL = (
    os.getenv("UPSTASH_REDIS_REST_URL", "").rstrip("/")
)
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")

WEATHER_SERVER_PATH = BASE_DIR / "custom_weather_mcp_server.py"
UVX_COMMAND = shutil.which("uvx") or "uvx"
logger = logging.getLogger(__name__)

if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
    logger.info(
        "Upstash cache is disabled because its REST URL or token is not configured."
    )


def _require_env(name: str, value: str | None) -> str:
    """Return an environment value or raise a readable setup error."""

    if not value:
        raise RuntimeError(
            f"{name} is missing. "
            f"Add {name}=your_key to the project .env file."
        )

    return value


def _subprocess_env(**updates: str | None) -> dict[str, str]:
    """
    Preserve the current Windows/Conda environment and add MCP API keys.
    """

    env = os.environ.copy()

    for key, value in updates.items():
        if value:
            env[key] = value

    return env


# =========================================================
# LLM
# =========================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=_require_env("GROQ_API_KEY", GROQ_API_KEY),
    max_tokens=128,
)


# =========================================================
# MCP client
# =========================================================

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": (
                "https://mcp.tavily.com/mcp/"
                f"?tavilyApiKey={TAVILY_API_KEY or ''}"
            ),
        },

        "aviationstack": {
            "transport": "stdio",
            "command": UVX_COMMAND,
            "args": [
                "aviationstack-mcp",
            ],
            "env": _subprocess_env(
                AVIATION_STACK_API_KEY=AVIATION_STACK_API_KEY,
            ),
        },

        "weather": {
            "transport": "stdio",

            # Uses the Python executable from the active Conda environment.
            "command": sys.executable,

            # Uses the weather server inside the current project folder.
            "args": [
                str(WEATHER_SERVER_PATH),
            ],

            "env": _subprocess_env(
                OPENWEATHER_API_KEY=OPENWEATHER_API_KEY,
            ),
        },
    }
)


async def _get_server_tool(
    server_name: str,
    tool_name: str,
):
    """
    Load one tool from one MCP server.

    This prevents a broken weather or AviationStack server from
    crashing an unrelated Tavily request.
    """

    if server_name == "tavily":
        _require_env(
            "TAVILY_API_KEY",
            TAVILY_API_KEY,
        )

    elif server_name == "aviationstack":
        _require_env(
            "AVIATION_STACK_API_KEY",
            AVIATION_STACK_API_KEY,
        )

        if shutil.which("uvx") is None:
            raise RuntimeError(
                "uvx was not found. Install uv, reopen the terminal, "
                "activate the travel environment, and run "
                "`uvx --version`."
            )

    elif server_name == "weather":
        _require_env(
            "OPENWEATHER_API_KEY",
            OPENWEATHER_API_KEY,
        )

        if not WEATHER_SERVER_PATH.is_file():
            raise FileNotFoundError(
                f"Weather MCP server not found: "
                f"{WEATHER_SERVER_PATH}"
            )

    # Important: load only the requested MCP server.
    tools = await client.get_tools(
        server_name=server_name,
    )

    tool = next(
        (
            item
            for item in tools
            if item.name == tool_name
        ),
        None,
    )

    if tool is None:
        available_tools = (
            ", ".join(
                sorted(item.name for item in tools)
            )
            or "none"
        )

        raise RuntimeError(
            f"MCP tool '{tool_name}' was not found "
            f"on server '{server_name}'. "
            f"Available tools: {available_tools}"
        )

    return tool


def _upstash_command(command: list[Any]) -> Any:
    response = requests.post(
        UPSTASH_REDIS_REST_URL,
        headers={
            "Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}",
            "Content-Type": "application/json",
        },
        json=command,
        timeout=(1.5, 2.5),
    )
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, dict) or "result" not in payload:
        raise ValueError("Unexpected response from Upstash Redis.")
    if payload.get("error"):
        raise RuntimeError("Upstash Redis returned a command error.")

    return payload["result"]


def _mcp_cache_key(
    server_name: str,
    tool_name: str,
    tool_args: dict[str, Any],
) -> str:
    key_input = json.dumps(
        tool_args,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    digest = hashlib.sha256(key_input.encode("utf-8")).hexdigest()
    return f"tripmate:mcp:v1:{server_name}:{tool_name}:{digest}"


def _is_provider_error_result(result: Any) -> bool:
    if isinstance(result, dict):
        return result.get("ok") is False

    if isinstance(result, list):
        for item in result:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                continue
            try:
                parsed_item = json.loads(item["text"])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed_item, dict) and parsed_item.get("ok") is False:
                return True

    return False


async def _cached_mcp_call(
    server_name: str,
    tool_name: str,
    tool_args: dict[str, Any],
    ttl_seconds: int,
) -> Any:
    cache_key = _mcp_cache_key(server_name, tool_name, tool_args)
    cache_enabled = bool(
        UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN
    )

    if cache_enabled:
        try:
            cached_value = await asyncio.to_thread(
                _upstash_command,
                ["GET", cache_key],
            )
            if cached_value is not None:
                if not isinstance(cached_value, str):
                    raise ValueError("Cached MCP result is not JSON text.")
                return json.loads(cached_value)
        except (
            requests.RequestException,
            RuntimeError,
            ValueError,
        ) as exc:
            logger.warning(
                "Upstash cache read failed (%s); calling the MCP provider.",
                type(exc).__name__,
            )

    tool = await _get_server_tool(server_name, tool_name)
    result = await tool.ainvoke(tool_args)

    if cache_enabled:
        if _is_provider_error_result(result):
            logger.warning(
                "MCP provider returned an error result; skipping cache write."
            )
            return result

        try:
            serialized_result = json.dumps(
                result,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            logger.warning(
                "MCP result is not JSON-serializable (%s); skipping cache write.",
                type(exc).__name__,
            )
        else:
            try:
                await asyncio.to_thread(
                    _upstash_command,
                    [
                        "SET",
                        cache_key,
                        serialized_result,
                        "EX",
                        str(ttl_seconds),
                    ],
                )
            except (
                requests.RequestException,
                RuntimeError,
                ValueError,
            ) as exc:
                logger.warning(
                    "Upstash cache write failed (%s); returning MCP result.",
                    type(exc).__name__,
                )

    return result


# =========================================================
# MCP connection test
# =========================================================

async def get_all_tools() -> None:
    """
    Test every MCP server independently.

    One failed server will not stop the remaining tests.
    """

    for server_name in (
        "tavily",
        "aviationstack",
        "weather",
    ):
        try:
            tools = await client.get_tools(
                server_name=server_name,
            )

            tool_names = (
                ", ".join(
                    tool.name
                    for tool in tools
                )
                or "no tools"
            )

            print(
                f"{server_name}: OK -> {tool_names}"
            )

        except Exception as exc:
            print(
                f"{server_name}: FAILED -> "
                f"{type(exc).__name__}: {exc}"
            )


# =========================================================
# Tavily MCP
# =========================================================

async def tavily_mcp_search(query: str):
    normalized_query = " ".join(query.split())
    return await _cached_mcp_call(
        "tavily",
        "tavily_search",
        {"query": normalized_query},
        ttl_seconds=900,
    )


# =========================================================
# AviationStack MCP
# =========================================================

async def aviation_mcp_call(
    tool_name: str,
    tool_args: dict[str, Any] | None = None,
):
    normalized_args = tool_args or {}
    ttl_seconds = (
        86400
        if tool_name in {"list_airports", "list_airlines"}
        else 300
    )
    return await _cached_mcp_call(
        "aviationstack",
        tool_name,
        normalized_args,
        ttl_seconds=ttl_seconds,
    )


# =========================================================
# Weather MCP
# =========================================================

async def weather_mcp_search(city: str):
    normalized_city = " ".join(city.split())
    return await _cached_mcp_call(
        "weather",
        "get_current_weather",
        {"city": normalized_city},
        ttl_seconds=600,
    )


async def forecast_mcp_search(city: str):
    normalized_city = " ".join(city.split())
    return await _cached_mcp_call(
        "weather",
        "get_forecast",
        {"city": normalized_city},
        ttl_seconds=3600,
    )


# =========================================================
# Destination extractor
# =========================================================

def extract_destination(query: str) -> str:
    prompt = f"""
Extract only the destination city or country from the travel request.

Travel request:
{query}

Return only the destination name.
Do not add any explanation.
"""

    response = llm.invoke(prompt)

    destination = str(
        response.content
    ).strip()

    if not destination:
        raise ValueError(
            "The destination could not be extracted."
        )

    return destination