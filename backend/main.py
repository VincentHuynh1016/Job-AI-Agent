from dotenv import load_dotenv
load_dotenv()
import asyncio
from mcp_server import initialize_mcp_server
from job_agents import run_analysist


async def main():
    server = await initialize_mcp_server()
    if not server:
        print("Failed to initialize MCP server")
        return

    linkedin_url = "https://www.linkedin.com/in/vincent-huynh-76729124b/"
    try:
        result = await run_analysist(server, linkedin_url)
        print(result)
    except Exception as e:
        print(f"Error during analysis: {e}")


if __name__ == "__main__":
    asyncio.run(main())
