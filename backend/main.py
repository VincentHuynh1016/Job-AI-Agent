import asyncio
from mcp_server import initialize_mcp_server
from job_agents import run_analysist
async def main():
    server = await initialize_mcp_server()
    linkedin_url = "https://www.linkedin.com/in/vincent-huynh-76729124b/"
    result = await run_analysist(server, linkedin_url)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
