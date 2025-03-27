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
    def __init__(self, llm_config: Dict):
        """Initialize the benchmark system with LLM configuration"""
        self.code_interpreter = CodeInterpreterTool()
        self.llm = LLM(**llm_config)
        
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

Make sure to handle all edge cases and follow Python best practices."""

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
                # Execute the test case
                exec(test_case, {'solution': solution_func})
                test_results.append(True)
            except Exception as e:
                test_results.append(False)
        
        # Clean up
        os.remove("temp_solution.py")
        
        return {
            'passed_tests': sum(test_results),
            'total_tests': len(test_results),
            'success_rate': sum(test_results) / len(test_results)
        }

    def clean_generated_code(self, code: str) -> str:
        """Clean the generated code by removing markdown formatting"""
        # Remove markdown code block markers
        code = code.replace('```python', '').replace('```', '')
        # Remove leading/trailing whitespace
        code = code.strip()
        return code

    def run_benchmark(self, num_problems: int = 10) -> List[Dict]:
        """Run the benchmark on a specified number of problems"""
        results = []
        
        for i, problem in enumerate(self.dataset['test']):
            if i >= num_problems:
                break
                
            print(f"\nProcessing problem {i+1}/{num_problems}")
            
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
                'execution_time': end_time - start_time,
                'evaluation': evaluation,
                'generated_code': generated_code
            })
            
        return results

    def save_results(self, results: List[Dict], filename: str = None):
        """Save benchmark results to a JSON file"""
        if filename is None:
            filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filename}")

def main():
    # Load environment variables
    load_dotenv()
    
    # Configure LLM (using Deepseek)
    llm_config = {
        "model": "deepseek/deepseek-chat",
        "api_key": os.getenv("DEEPSEEK_API_KEY")
    }
    
    # Initialize benchmark system
    benchmark = BenchmarkSystem(llm_config)
    
    # Run benchmark on 5 problems
    results = benchmark.run_benchmark(num_problems=5)
    
    # Save results
    benchmark.save_results(results)
    
    # Print summary
    total_problems = len(results)
    total_tests = sum(r['evaluation']['total_tests'] for r in results)
    passed_tests = sum(r['evaluation']['passed_tests'] for r in results)
    avg_time = sum(r['execution_time'] for r in results) / total_problems
    
    print("\nBenchmark Summary:")
    print(f"Total Problems: {total_problems}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed Tests: {passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.2f}%")
    print(f"Average Execution Time: {avg_time:.2f} seconds")

if __name__ == "__main__":
    main() 