import os
import gradio as gr
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import CodeInterpreterTool
from dotenv import load_dotenv
from typing import Any, List, Dict
import json
from datetime import datetime

# Load environment variables
load_dotenv()

class AutonomousCodingAgent:
    def __init__(self, model_config: Dict):
        """Initialize the autonomous coding agent"""
        self.code_interpreter = CodeInterpreterTool()
        self.llm = LLM(**model_config)
        self.model_name = model_config.get('model', 'unknown')
        
        self.programmer_agent = Agent(
            role="Python Programmer",
            goal="Write and execute Python code to solve problems",
            backstory="An expert Python programmer who can write efficient code to solve complex problems.",
            tools=[self.code_interpreter],
            verbose=True,
            llm=self.llm
        )

    def solve_coding_task(self, task_description: str, expected_output: str) -> Dict:
        """Solve a coding task and return results"""
        task = Task(
            description=task_description,
            expected_output=expected_output,
            agent=self.programmer_agent
        )

        crew = Crew(
            agents=[self.programmer_agent],
            tasks=[task],
            verbose=True,
            process=Process.sequential
        )

        # Execute the task
        result = crew.kickoff()
        
        # Clean and format the code
        code = self.clean_generated_code(str(result))
        
        return {
            'code': code,
            'model': self.model_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def clean_generated_code(self, code: str) -> str:
        """Clean the generated code by removing markdown formatting and extracting only the Python code"""
        # Remove markdown code block markers
        code = code.replace('```python', '```').strip()
        
        # Extract code between ``` markers if present
        if '```' in code:
            code_blocks = code.split('```')
            # Take the first code block that appears between ``` markers
            for block in code_blocks:
                block = block.strip()
                if block and not block.lower().startswith(('here', 'this', 'the', 'i ', 'first')):
                    return block
        
        # If no code blocks found, try to find the first line that looks like Python code
        lines = code.split('\n')
        for line in lines:
            line = line.strip()
            # Look for common Python code starters
            if line.startswith(('def ', 'class ', 'import ', 'from ', '#')):
                # Include this line and all subsequent lines
                start_idx = lines.index(line)
                return '\n'.join(lines[start_idx:]).strip()
        
        return code.strip()

def create_agent(model_name: str) -> AutonomousCodingAgent:
    """Create an agent with the specified model configuration"""
    if model_name == "deepseek":
        config = {
            "model": "deepseek/deepseek-chat",
            "api_key": os.getenv("DEEPSEEK_API_KEY")
        }
    elif model_name == "gemini":
        config = {
            "model": "gemini/gemini-2.5-pro-exp-03-25",
            "temperature": 0.1,
            "api_key": os.getenv("GEMINI_API_KEY")
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return AutonomousCodingAgent(config)

def generate_code(model_name: str, task_points: str, expected_output: str) -> Dict:
    """Generate code using the specified model"""
    agent = create_agent(model_name)
    
    # Format task description
    task_points_list = [f"{i+1}. {point.strip()}" for i, point in enumerate(task_points.split('\n')) if point.strip()]
    task_description = "Write Python code to:\n" + "\n".join(task_points_list)
    
    # Generate solution
    result = agent.solve_coding_task(task_description, expected_output)
    
    # Save result
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"solution_{model_name}_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result

def create_ui():
    """Create the Gradio UI"""
    with gr.Blocks(title="Autonomous Coding Agent", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🤖 Autonomous Coding Agent
        
        This tool uses AI to generate Python code based on your requirements.
        
        ### How to use:
        1. Select the AI model to use
        2. Enter your coding requirements (one per line)
        3. Specify the expected output
        4. Click Generate to create the code
        """)
        
        with gr.Row():
            with gr.Column():
                model = gr.Radio(
                    choices=["deepseek", "gemini"],
                    label="Select Model",
                    value="deepseek"
                )
                
                task = gr.Textbox(
                    label="Coding Requirements",
                    placeholder="Enter each requirement on a new line...",
                    lines=5
                )
                
                output = gr.Textbox(
                    label="Expected Output",
                    placeholder="Describe what the code should output..."
                )
                
                generate_btn = gr.Button("🚀 Generate Code", variant="primary")
            
            with gr.Column():
                code_output = gr.Code(
                    label="Generated Code",
                    language="python"
                )
                
                with gr.Accordion("Solution Details", open=False):
                    timestamp = gr.Textbox(label="Timestamp")
                    used_model = gr.Textbox(label="Model Used")
        
        def process_request(model_name, task_points, expected_output):
            try:
                result = generate_code(model_name, task_points, expected_output)
                return [
                    result['code'],
                    result['timestamp'],
                    result['model']
                ]
            except Exception as e:
                return [
                    f"Error: {str(e)}",
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    model_name
                ]
        
        generate_btn.click(
            process_request,
            inputs=[model, task, output],
            outputs=[code_output, timestamp, used_model]
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["deepseek", "Create a function to calculate fibonacci sequence\nMake it efficient using memoization", "Returns fibonacci number for given input"],
                ["gemini", "Create a function to sort a list\nImplement quicksort algorithm", "Returns sorted list"]
            ],
            inputs=[model, task, output]
        )
    
    return app

if __name__ == "__main__":
    app = create_ui()
    app.launch(share=True)