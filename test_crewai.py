import os
from crewai import Agent, Task, Crew
from dotenv import load_dotenv

load_dotenv()


# Importing crewAI tools
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
)


# Instantiate tools
docs_tool = DirectoryReadTool(directory="./blog-posts")
file_tool = FileReadTool()


writer = Agent(
    role="Content Writer",
    goal="Craft engaging blog posts about the AI industry",
    backstory="A skilled writer with a passion for technology.",
    tools=[docs_tool, file_tool],
    verbose=True,
)


write = Task(
    description="Write an engaging blog post about the AI industry, based on the research analysts summary. Draw inspiration from the latest blog posts in the directory.",
    expected_output="A 4-paragraph blog post formatted in markdown with engaging, informative, and accessible content, avoiding complex jargon.",
    agent=writer,
    output_file="blog-posts/new_post.md",  # The final blog post will be saved here
)

# Assemble a crew with planning enabled
crew = Crew(
    agents=[writer],
    tasks=[write],
    verbose=True,
    planning=True,  # Enable planning feature
)

# Execute tasks
response = crew.kickoff()
breakpoint()
