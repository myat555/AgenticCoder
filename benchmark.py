import os
from datasets import load_dataset
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import CodeInterpreterTool
from dotenv import load_dotenv
from typing import Any, List, Dict
import json
import time
from datetime import datetime
import importlib.util


# Custom tool to execute code
class CodeInterpreterTool(CodeInterpreterTool):
    def _run(self, code: str, libraries_used: list[str] | None = None) -> Any:
        """Execute code and return the result"""
        return super()._run(code=code, libraries_used=libraries_used or [])

class BenchmarkSystem:
    def __init__(self, llm_config: Dict, model_name: str):
        """Initialize the benchmark system with LLM configuration"""
        self.code_interpreter = CodeInterpreterTool()
        self.llm = LLM(**llm_config)
        self.model_name = model_name
        
        self.programmer_agent = Agent(
            role="Python Programmer",
            goal="Write and execute Python code to solve problems",
            backstory="An expert Python programmer who can write efficient code to solve complex problems.",
            tools=[self.code_interpreter],
            verbose=True,
            llm=self.llm
        )
        
        # Load HumanEval dataset
        self.dataset = load_dataset("openai/openai_humaneval")
        
    def create_coding_task(self, problem: Dict) -> Task:
        """Create a coding task from a HumanEval problem"""
        # Extract function signature more safely
        canonical_solution = problem['canonical_solution']
        function_signature = canonical_solution.split('\n')[0] if canonical_solution else "def function_name()"
        
        prompt = f"""Write a Python function that solves the following problem:

{problem['prompt']}

The function should be named {problem['entry_point']} and should match the following signature:
{function_signature}

Important:
1. Include all necessary imports at the top of the code
2. Handle all edge cases
3. Follow Python best practices
4. Make sure the function is self-contained and doesn't rely on external variables
5. Return the correct type as specified in the function signature"""

        return Task(
            description=prompt,
            expected_output="Function should pass all test cases",
            agent=self.programmer_agent,
        )

    def evaluate_solution(self, problem: Dict, generated_code: str) -> Dict:
        """Evaluate the generated solution against test cases"""
        # Create a temporary file with the generated code
        with open("temp_solution.py", "w") as f:
            f.write(generated_code)
        
        # Import the generated function using importlib
        spec = importlib.util.spec_from_file_location("temp_solution", "temp_solution.py")
        temp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(temp_module)
        solution_func = getattr(temp_module, problem['entry_point'])
        
        # Run test cases
        test_results = []
        for test_case in problem['test']:
            try:
                exec(test_case, {'solution': solution_func})
                test_results.append(True)
            except Exception:
                test_results.append(False)
        
        # Clean up
        os.remove("temp_solution.py")
        
        return {
            'passed_tests': sum(test_results),
            'total_tests': len(test_results),
            'success_rate': sum(test_results) / len(test_results)
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

    def run_benchmark(self, num_problems: int = 10) -> List[Dict]:
        """Run the benchmark on a specified number of problems"""
        results = []
        
        for i, problem in enumerate(self.dataset['test']):
            if i >= num_problems:
                break
                
            print(f"\n{'='*50}")
            print(f"[{self.model_name}] Processing problem {i+1}/{num_problems}")
            print(f"Function: {problem['entry_point']}")
            print(f"{'='*50}")
            
            # Create and run the task
            task = self.create_coding_task(problem)
            crew = Crew(
                agents=[self.programmer_agent],
                tasks=[task],
                verbose=True,
                process=Process.sequential,
            )
            
            start_time = time.time()
            crew_output = crew.kickoff()
            end_time = time.time()
            
            # Extract and clean the code from the crew output
            generated_code = self.clean_generated_code(str(crew_output))
            
            # Evaluate the solution
            evaluation = self.evaluate_solution(problem, generated_code)
            
            results.append({
                'problem_id': i,
                'task_id': problem['task_id'],
                'function_name': problem['entry_point'],
                'execution_time': end_time - start_time,
                'evaluation': evaluation,
                'generated_code': generated_code
            })
            
            # Print problem summary
            print(f"Success Rate: {evaluation['success_rate']*100:.2f}%")
            print(f"Execution Time: {end_time - start_time:.2f} seconds")
            print(f"{'='*50}")
            
        return results

    def save_results(self, results: List[Dict], filename: str = None):
        """Save benchmark results to a JSON file"""
        if filename is None:
            filename = f"benchmark_results_{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filename}")

def print_comparison_summary(deepseek_results: List[Dict], gemini_results: List[Dict]):
    """Print a comparison summary of both models"""
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    
    def calculate_metrics(results):
        total_problems = len(results)
        total_tests = sum(r['evaluation']['total_tests'] for r in results)
        passed_tests = sum(r['evaluation']['passed_tests'] for r in results)
        avg_time = sum(r['execution_time'] for r in results) / total_problems
        success_rate = passed_tests / total_tests * 100
        return total_problems, total_tests, passed_tests, avg_time, success_rate
    
    ds_metrics = calculate_metrics(deepseek_results)
    gm_metrics = calculate_metrics(gemini_results)
    
    headers = ["Metric", "Deepseek", "Gemini"]
    metrics = [
        ["Total Problems", str(ds_metrics[0]), str(gm_metrics[0])],
        ["Total Tests", str(ds_metrics[1]), str(gm_metrics[1])],
        ["Passed Tests", str(ds_metrics[2]), str(gm_metrics[2])],
        ["Success Rate", f"{ds_metrics[4]:.2f}%", f"{gm_metrics[4]:.2f}%"],
        ["Avg Execution Time", f"{ds_metrics[3]:.2f}s", f"{gm_metrics[3]:.2f}s"]
    ]
    
    # Calculate column widths
    widths = [max(len(str(row[i])) for row in [headers] + metrics) for i in range(3)]
    
    # Print headers
    print("".join(f"{headers[i]:<{widths[i]+4}}" for i in range(3)))
    print("-" * (sum(widths) + 8))
    
    # Print metrics
    for row in metrics:
        print("".join(f"{str(row[i]):<{widths[i]+4}}" for i in range(3)))
    
    print("="*70)

def main():
    # Load environment variables
    load_dotenv()
    
    # Number of problems to test
    num_problems = 5
    
    # Configure Deepseek
    deepseek_config = {
        "model": "deepseek/deepseek-chat",
        "api_key": os.getenv("DEEPSEEK_API_KEY")
    }
    
    # Configure Gemini
    gemini_config = {
        "model": "gemini/gemini-2.5-pro-exp-03-25",
        "temperature": 0.1,
        "api_key": os.getenv("GEMINI_API_KEY")
    }
    
    # Run Deepseek benchmark
    print("\nRunning Deepseek benchmark...")
    deepseek_benchmark = BenchmarkSystem(deepseek_config, "Deepseek")
    deepseek_results = deepseek_benchmark.run_benchmark(num_problems=num_problems)
    deepseek_benchmark.save_results(deepseek_results)
    
    # Run Gemini benchmark
    print("\nRunning Gemini benchmark...")
    gemini_benchmark = BenchmarkSystem(gemini_config, "Gemini")
    gemini_results = gemini_benchmark.run_benchmark(num_problems=num_problems)
    gemini_benchmark.save_results(gemini_results)
    
    # Print comparison summary
    print_comparison_summary(deepseek_results, gemini_results)
    
    # Save combined results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_results = {
        'timestamp': timestamp,
        'num_problems': num_problems,
        'models': {
            'deepseek': {
                'config': deepseek_config,
                'results': deepseek_results
            },
            'gemini': {
                'config': gemini_config,
                'results': gemini_results
            }
        }
    }
    
    with open(f"benchmark_comparison_{timestamp}.json", 'w') as f:
        json.dump(combined_results, f, indent=2)
    print(f"\nCombined results saved to benchmark_comparison_{timestamp}.json")

if __name__ == "__main__":
    main() 