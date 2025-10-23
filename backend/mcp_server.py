"""Initializing/Setting up our MCP server."""
import os
import logging
import asyncio
from agents.mcp import MCPServerStdio

logger = logging.getLogger(__name__)
_mcp_server = None

# We will now create a initialize function for our MCP Server
async def initialize_mcp_server():
    """Initializing our MCP server."""
    global _mcp_server

    if _mcp_server:
        return _mcp_server

    try:
        server = MCPServerStdio(
            cache_tools_list = False,
            params={
                "command":"npx",
                "args": ["-y", "@brightdata/mcp"], 
                "env": {
                    "API_TOKEN": os.environ["BRIGHT_DATA_API_KEY"],
                    "WEB_UNLOCKER_ZONE": os.environ["WEB_UNLOCKER_ZONE"]
                }
            }
        )
        # Starting the MCP server process
        await asyncio.wait_for(server.__aenter__(), timeout=10)
        _mcp_server = server
        return server

    except Exception as e:
        logger.error(f"Error when initializing MCP server: {e}")
        return None
    
"""
This function does a few important things: 
1. Prevents re-iniitalizatoin if the server is already up
2. Spins up a new MCP server instance
3. Handles timeout and logs any failures gracefully
"""

#Helper functions
async def wait_for_initialization():
    """Wait for MCP initialization to complete."""
    return await initialize_mcp_server() is not None

def get_mcp_server():
    "Get the current MCP server instance."""
    return _mcp_server
