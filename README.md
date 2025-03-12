# AgenticCoder
## Agentic Code Development and Testing with crewAI

This project aims to use CrewAI, an open-source framework for building AI agents and multi-agent systems, to create an autonomous crew of agents capable of writing, running, and testing code in Python and other supported programming languages. The system will try to automate the full development workflow—from generating code based on user input to executing it and verifying its correctness through dynamic testing. The LLMs being utilized will either be Ollama models running locally or other LLM access through API.

## Requirements

- Python 3.9 or higher
- Install dependencies: `pip install crewai ollama`
- Set up Ollama (if using local models): Follow installation instructions at [Ollama.ai](https://ollama.ai)
- API keys for any cloud-based LLM services configured in environment variables
- Docker is required in order to run the CodeInterpreterTool that will execute the code in a container
