#!/usr/bin/env python
import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import CodeInterpreterTool
from dotenv import load_dotenv

# Initialize the tool
code_interpreter = CodeInterpreterTool()

# Load environment variables from .env file
load_dotenv()

deepseek_llm = LLM(
    model="deepseek/deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# Define an agent that uses the tool
programmer_agent = Agent(
    role="Python Programmer",
    goal="Write and execute Python code to solve problems",
    backstory="An expert Python programmer who can write efficient code to solve complex problems.",
    tools=[code_interpreter],
    verbose=True,
    llm=deepseek_llm
)

# Get user input for task description and expected output
print("Write Python code to:")
task_points = []
count = 1
while True:
    point = input(f"{count}. ").strip()
    if not point:
        print("Task description cannot be empty. Please enter a valid point.")
        continue
    task_points.append(f"{count}. {point}")
    count += 1
    
    while True:
        more = input("Do you want to add more points? (y/n): ").strip().lower()
        if more in ["y", "n"]:
            break
        print("Invalid input. Please enter 'y' or 'n'.")
    
    if more == "n":
        break

task_description = "Write Python code to:\n" + "\n".join(task_points)

expected_output = input("Enter the expected output: ").strip()
while not expected_output:
    print("Expected output cannot be empty. Please enter a valid expected output.")
    expected_output = input("Enter the expected output: ").strip()

# Example task to generate and execute code
coding_task = Task(
    description=task_description,
    expected_output=expected_output,
    agent=programmer_agent,
)

# Create and run the crew
crew = Crew(
    agents=[programmer_agent],
    tasks=[coding_task],
    verbose=True,
    process=Process.sequential,
)


result = crew.kickoff()