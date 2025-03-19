import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import CodeInterpreterTool
from dotenv import load_dotenv
from typing import Any

# Custom tool to save generated code
class CodeSaverTool(CodeInterpreterTool):
    def _run(self, code: str, libraries_used: list[str] | None = None) -> Any:
        """Execute code first, then save to descriptive filename"""
        # Execute code first
        result = super()._run(code=code, libraries_used=libraries_used or [])
        filename = f"generated_code.py"
        
        # Save code after successful execution
        with open(filename, "w") as f:
            f.write(code)
        return result

# Initialize the tool
code_interpreter = CodeSaverTool()

# Load environment variables from .env file
load_dotenv()

deepseek_llm = LLM(
    model="deepseek/deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# For Ollama LLM, you need to run the Ollama server locally
ollama_llm = LLM(
    model="ollama/qwen2.5-coder:7b",
    base_url="http://localhost:11434"
)

# Define an agent that uses the tool
programmer_agent = Agent(
    role="Python Programmer",
    goal="Write and execute Python code to solve problems",
    backstory="An expert Python programmer who can write efficient code to solve complex problems.",
    tools=[code_interpreter],
    verbose=True,
    llm=ollama_llm
    # To use Ollama LLM, change the llm to ollama_llm
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

task_description = "Write Python code to:\n" + "\n".join(task_points) + "\n\n" + "Make sure to handle any necessary imports and print the results."

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