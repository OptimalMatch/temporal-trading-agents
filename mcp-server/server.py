"""
MCP (Model Context Protocol) Server for Temporal Trading Agents.
Allows AI agents like Claude to access trading strategies through standardized protocol.
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import asyncio
import json
from typing import Any, Dict, List
from mcp.server import Server
from mcp.types import Tool, TextContent
import httpx

# Backend API URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

# Initialize MCP server
server = Server("temporal-trading-agents")


# ==================== Tools ====================

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List all available MCP tools"""
    return [
        Tool(
            name="analyze_gradient_strategy",
            description="Run forecast gradient strategy analysis for a trading symbol. Analyzes the SHAPE of the forecast curve to determine entry/exit timing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading symbol (e.g., BTC-USD, AAPL, ETH-USD)"
                    }
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="analyze_confidence_strategy",
            description="Run confidence-weighted strategy using model agreement to determine position sizing. Uses ensemble consensus for risk management.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading symbol (e.g., BTC-USD, AAPL, ETH-USD)"
                    }
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="analyze_all_strategies",
            description="Run ALL 8 advanced trading strategies and return consensus analysis. Provides comprehensive view with STRONG BUY/SELL/MIXED signals.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading symbol (e.g., BTC-USD, AAPL, ETH-USD)"
                    },
                    "horizons": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Forecast horizons in days (default: [3, 7, 14, 21])",
                        "default": [3, 7, 14, 21]
                    }
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_analysis_history",
            description="Get historical strategy analyses for a symbol. Useful for tracking strategy performance over time.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading symbol"
                    },
                    "strategy_type": {
                        "type": "string",
                        "enum": ["gradient", "confidence", "timeframe", "volatility", "mean_reversion", "acceleration", "swing", "risk_adjusted"],
                        "description": "Filter by specific strategy type (optional)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10
                    }
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_consensus_history",
            description="Get historical consensus results for a symbol. Shows how consensus has changed over time.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading symbol"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10
                    }
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_symbol_analytics",
            description="Get comprehensive analytics for a trading symbol including total analyses, strategy breakdown, and latest consensus.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading symbol"
                    }
                },
                "required": ["symbol"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls from agents"""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            if name == "analyze_gradient_strategy":
                response = await client.post(
                    f"{BACKEND_URL}/api/v1/analyze/gradient",
                    params={"symbol": arguments["symbol"]}
                )
                response.raise_for_status()
                result = response.json()

                return [TextContent(
                    type="text",
                    text=format_strategy_result(result, "Gradient")
                )]

            elif name == "analyze_confidence_strategy":
                response = await client.post(
                    f"{BACKEND_URL}/api/v1/analyze/confidence",
                    params={"symbol": arguments["symbol"]}
                )
                response.raise_for_status()
                result = response.json()

                return [TextContent(
                    type="text",
                    text=format_strategy_result(result, "Confidence-Weighted")
                )]

            elif name == "analyze_all_strategies":
                request_data = {
                    "symbol": arguments["symbol"],
                    "horizons": arguments.get("horizons", [3, 7, 14, 21])
                }
                response = await client.post(
                    f"{BACKEND_URL}/api/v1/analyze/consensus",
                    json=request_data
                )
                response.raise_for_status()
                result = response.json()

                return [TextContent(
                    type="text",
                    text=format_consensus_result(result)
                )]

            elif name == "get_analysis_history":
                params = {
                    "symbol": arguments["symbol"],
                    "limit": arguments.get("limit", 10)
                }
                if "strategy_type" in arguments:
                    params["strategy_type"] = arguments["strategy_type"]

                response = await client.get(
                    f"{BACKEND_URL}/api/v1/history/analyses/{arguments['symbol']}",
                    params=params
                )
                response.raise_for_status()
                result = response.json()

                return [TextContent(
                    type="text",
                    text=format_history(result, "analyses")
                )]

            elif name == "get_consensus_history":
                response = await client.get(
                    f"{BACKEND_URL}/api/v1/history/consensus/{arguments['symbol']}",
                    params={"limit": arguments.get("limit", 10)}
                )
                response.raise_for_status()
                result = response.json()

                return [TextContent(
                    type="text",
                    text=format_history(result, "consensus")
                )]

            elif name == "get_symbol_analytics":
                response = await client.get(
                    f"{BACKEND_URL}/api/v1/analytics/{arguments['symbol']}"
                )
                response.raise_for_status()
                result = response.json()

                return [TextContent(
                    type="text",
                    text=format_analytics(result)
                )]

            else:
                return [TextContent(
                    type="text",
                    text=f"Unknown tool: {name}"
                )]

    except httpx.HTTPError as e:
        return [TextContent(
            type="text",
            text=f"Error calling backend API: {str(e)}"
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error executing tool: {str(e)}"
        )]


# ==================== Formatting Functions ====================

def format_strategy_result(result: Dict, strategy_name: str) -> str:
    """Format strategy result for agent consumption"""
    signal = result.get("signal", {})

    output = f"""
# {strategy_name} Strategy Analysis

**Symbol:** {result.get('symbol')}
**Current Price:** ${result.get('current_price', 0):,.2f}

## Signal
- **Action:** {signal.get('signal', 'UNKNOWN')}
- **Position Size:** {signal.get('position_size_pct', 0):.0f}%
- **Confidence:** {signal.get('confidence', 'N/A')}

## Details
"""

    if signal.get('target_price'):
        output += f"- **Target Price:** ${signal['target_price']:,.2f}\n"
    if signal.get('stop_loss'):
        output += f"- **Stop Loss:** ${signal['stop_loss']:,.2f}\n"
    if signal.get('rationale'):
        output += f"- **Rationale:** {signal['rationale']}\n"

    if result.get('execution_time_ms'):
        output += f"\n**Execution Time:** {result['execution_time_ms']}ms\n"

    return output


def format_consensus_result(result: Dict) -> str:
    """Format consensus result for agent consumption"""
    output = f"""
# Comprehensive 8-Strategy Consensus Analysis

**Symbol:** {result.get('symbol')}
**Current Price:** ${result.get('current_price', 0):,.2f}

## Consensus
- **Overall Signal:** {result.get('consensus')}
- **Strength:** {result.get('strength')}
- **Average Position Size:** {result.get('avg_position', 0):.0f}%

## Strategy Breakdown
- **Bullish:** {result.get('bullish_count')}/{result.get('total_count')}
  {format_strategy_list(result.get('bullish_strategies', []))}

- **Bearish/Neutral:** {result.get('bearish_count') + result.get('neutral_count')}/{result.get('total_count')}
  {format_strategy_list(result.get('bearish_strategies', []) + result.get('neutral_strategies', []))}

## Individual Strategy Signals
"""

    strategies = result.get('strategies', {})
    for name, signal_data in strategies.items():
        signal = signal_data.get('signal', 'UNKNOWN')
        position = signal_data.get('position_size_pct', 0)
        output += f"- **{name}:** {signal} ({position:.0f}%)\n"

    output += f"\n**Analysis Time:** {result.get('analyzed_at', 'N/A')}\n"

    return output


def format_strategy_list(strategies: List[str]) -> str:
    """Format list of strategy names"""
    if not strategies:
        return "(none)"
    return "\n  ".join([f"- {s}" for s in strategies])


def format_history(result: Dict, history_type: str) -> str:
    """Format historical data"""
    symbol = result.get('symbol')
    count = result.get('count', 0)

    output = f"""
# Historical {history_type.title()} for {symbol}

**Total Records:** {count}

"""

    items = result.get('analyses', []) if history_type == "analyses" else result.get('results', [])

    for idx, item in enumerate(items[:10], 1):  # Show first 10
        if history_type == "analyses":
            signal = item.get('signal', {})
            output += f"""
## {idx}. {item.get('strategy_type', 'Unknown').upper()} - {item.get('created_at', 'N/A')}
- **Signal:** {signal.get('signal', 'UNKNOWN')}
- **Position:** {signal.get('position_size_pct', 0):.0f}%
- **Price:** ${item.get('current_price', 0):,.2f}
"""
        else:  # consensus
            output += f"""
## {idx}. {item.get('created_at', 'N/A')}
- **Consensus:** {item.get('consensus', 'UNKNOWN')}
- **Strength:** {item.get('strength', 'N/A')}
- **Price:** ${item.get('current_price', 0):,.2f}
- **Bullish:** {item.get('bullish_count')}/{item.get('total_count')}
"""

    return output


def format_analytics(result: Dict) -> str:
    """Format analytics data"""
    return f"""
# Analytics for {result.get('symbol')}

## Overview
- **Total Analyses:** {result.get('total_analyses', 0)}
- **Consensus Analyses:** {result.get('consensus_count', 0)}

## Strategy Breakdown
{format_strategy_breakdown(result.get('strategy_breakdown', {}))}

## Latest Consensus
{format_latest_consensus(result.get('latest_consensus'))}
"""


def format_strategy_breakdown(breakdown: Dict) -> str:
    """Format strategy breakdown"""
    if not breakdown:
        return "(No data)"

    output = ""
    for strategy, count in breakdown.items():
        output += f"- **{strategy.title()}:** {count} analyses\n"
    return output


def format_latest_consensus(consensus: Dict) -> str:
    """Format latest consensus"""
    if not consensus:
        return "(No consensus data available)"

    return f"""
- **Signal:** {consensus.get('consensus', 'N/A')}
- **Strength:** {consensus.get('strength', 'N/A')}
- **Price:** ${consensus.get('current_price', 0):,.2f}
- **Bullish:** {consensus.get('bullish_count')}/{consensus.get('total_count')}
- **Time:** {consensus.get('created_at', 'N/A')}
"""


# ==================== Main ====================

async def main():
    """Main entry point"""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        print("🤖 MCP Server: Starting Temporal Trading Agents MCP Server", file=sys.stderr)
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
