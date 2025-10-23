"""Agents for analyzing LinkedIn profiles and suggesting jobs."""
import os
import logging  # Logging modules for logging info
import asyncio  # For asynchronous programming
from agents import (
    Agent,  # Base agent class
    Runner,  # Running/executing the agent
    OpenAIChatCompletionsModel,  # Initialize the OpenAI chat model
    set_tracing_disabled,  # Disable the progress logs that openAI has
)
from agents.mcp import MCPServer  # Model Context Protocol server
from openai import AsyncOpenAI  # Aync to open OpenAI's client

logger = logging.getLogger(__name__)

# API KEY FOR OPENAI
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def linkedIn_Agent(mcp_server: MCPServer) -> Agent:
    """This will be our linkedin analyzer agent

    This agent will examine your linkedin profile and extract relevant career insights

    Args:
        mcp_server (MCPServer): The MCP server instance to use for the agent.
        linkedin_url (str): The LinkedIn profile URL to analyze.
    """
    set_tracing_disabled(disabled=True)
    # Initialize the OpenAI client
    client = AsyncOpenAI(OPENAI_API_KEY)

    linkedIn_agent = Agent(
        name="LinkedIn Profile Analyzer",
        instructions=f"""You are a LinkedIn profile analyzer. 
        Analyze profiles for :

        - Professional experieince and career pogression
        - Education and certifications
        - Core Skills and expertise
        - Current role and Company
        - Previous roles and achievements
        - Industry reputations (recommendations/endorsements)

        Provide a structured analysis with bullet points and a brief executive summary.
        """,
        mcp_server=[mcp_server],
        model=OpenAIChatCompletionsModel(model="gpt-5", openai_client=client),
    )
    return linkedIn_agent


def job_suggestion_agent() -> Agent:
    """This agent will suggest the preferred domain

    Based on the Linkedin analysis it will suggest the preferred job
    """
    set_tracing_disabled(disabled=True)
    client = AsyncOpenAI(OPENAI_API_KEY)
    job_suggesting_agent = Agent(
        name="Job Suggestions",
        instructions=f""" You are a domain classifier that identifies the 
        primary profession domain from a LinkedIn profile. 
        """,
        model=OpenAIChatCompletionsModel(model="gpt-5", openai_client=client),
    )
    return job_suggesting_agent


def url_generator_agent() -> Agent:
    """This agent will generate URL's

    This agent takes the user’s preferred job domain (e.g., AI, healthcare, fintech)
    and generates the corresponding Y Combinator job board URL to display relevant
    companies and roles.
    """
    set_tracing_disabled(disabled=True)
    client = AsyncOpenAI(OPENAI_API_KEY)
    url_generator_agent = Agent(
        name="URL Generator",
        instructions=f"""You are a URL generator that creates Y Combinator job board URLs based on domains.
        """,
        model=OpenAIChatCompletionsModel(model="gpt-5", openai_client=client),
    )

    return url_generator_agent


def job_search_agent(mcp_server: MCPServer) -> Agent:
    """Generate URL and pull out real job listings

    This agent will take the generated URL and scrape real job listings from
    Y Combinator's job board.
    """
    set_tracing_disabled(disabled=True)
    client = AsyncOpenAI(OPENAI_API_KEY)
    job_searching_agent = Agent(
        name="Job Finder",
        instructions=f""" You are a job finder that extracts job listings from Y Combinator's job board.
        """,
        mcp_servers=[mcp_server],
        model=OpenAIChatCompletionsModel(model="gpt-5", openai_client=client),
    )

    return job_searching_agent


def url_parser_agent() -> Agent:
    """This agent will parse URLs

    Sometimes YC job links are behind authntication redirects. This agent will cleans
    up and simplifies those URL's
    """
    set_tracing_disabled(disabled=True)
    client = AsyncOpenAI(OPENAI_API_KEY)
    url_parseing_agent = Agent(
        name="URL Parser",
        instructions=f""" You are a URL parser that transforms Y Combinator authentication URLs into direct job URLS.
        """,
        model=OpenAIChatCompletionsModel(model="gpt-5", openai_client=client),
    )
    return url_parseing_agent


def summary_agent() -> Agent:
    """This agent will summarize everything

    The agent will summarize the user's profile, the domain prediction, and the job
    results into a clean markdown format
    """
    set_tracing_disabled(disabled=True)
    client  = AsyncOpenAI(OPENAI_API_KEY)
    summarization_agent = Agent(
        name="Summary Agent",
        instructions=f"""You are a summary agent that creates comprehensive career analysis reports.

        Ensure your response is well-formatted markdown that can be directly displayed.""",
        model=OpenAIChatCompletionsModel(model="gpt-5", openai_client=client),
    )
    return summarization_agent


#I was thinking if I were to make this a flask it would be a GET request
async def run_analysist(mcp_server: MCPServer, linkedin_url: str):
    """Run the entire analysis pipeline"""  

    # Start linkedin profile analysis
    logger.info("Running the LinkedIN profile analysis")
    linkedIn_results = await Runner.run(starting_agent=linkedIn_Agent(mcp_server), input=linkedin_url)
    logger.info("Linkedin profile analysis has been completed")

    # Get job suggestions
    logger.info("Getting Job Suggestions")
    suggestion_results = await Runner.run(starting_agent=job_suggestion_agent(), input=linkedIn_results)
    logger.info("Job suggestions completed")

    # Generate URL's
    logger.info("Generating Job link")
    job_link_results = await Runner.run(starting_agent=url_generator_agent(), input=suggestion_results)
    logger.info("Job link generation completed")

    # Get job matches
    logger.info("Getting Job matches")
    job_search_results = await Runner.run(starting_agent=job_search_agent(mcp_server), input=job_link_results)
    logger.info("Job search completed")

    # Parse URLs to get direct job links
    logger.info("Parsing Job URLs")
    job_search_results = await Runner.run(starting_agent=url_parser_agent(), input=job_search_results)
    logger.info("URL parsing completed")

    # Create a single input for the summary agent
    logger.info("Generating final summary")
    summary_input = f"""LinkedIn Profile Analysis:
    {linkedIn_results.final_output}

    Job Suggestions:
    {suggestion_results.final_output}

    Job Matches:
    {job_search_results.final_output}

    Please analyze the above information and create a comprehensive career analysis report in markdown format."""

    #Get final summary with a single call
    summary_result = await Runner.run(starting_agent=summary_agent(), input=summary_input)
    logger.info("Summary generation completed")

    return summary_result.final_output



