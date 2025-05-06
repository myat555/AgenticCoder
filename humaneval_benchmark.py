import os
import re
import json
import time
import importlib.util
from datetime import datetime
from typing import Any, List, Dict
from datasets import load_dataset
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import CodeInterpreterTool
from dotenv import load_dotenv


class CodeInterpreterTool(CodeInterpreterTool):
    def _run(self, code: str, libraries_used: list[str] | None = None) -> Any:
        return super()._run(code=code, libraries_used=libraries_used or [])


class BenchmarkSystem:
    def __init__(self, llm_config: Dict, model_name: str):
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
        self.dataset = load_dataset("openai/openai_humaneval")

    def create_coding_task(self, problem: Dict) -> Task:
        function_signature = problem['canonical_solution'].split('\n')[0] if problem['canonical_solution'] else "def function_name()"

        prompt = f"""Write a Python function that solves the following problem:

{problem['prompt']}

The function should be named `{problem['entry_point']}` and follow this signature:
{function_signature}

Important:
1. Include necessary imports
2. Handle edge cases and match expected behavior
3. Use idiomatic Python
4. Ensure the code runs independently
"""

        return Task(
            description=prompt,
            expected_output="Function should pass all test cases",
            agent=self.programmer_agent,
        )

    def evaluate_solution(self, problem: Dict, generated_code: str) -> Dict:
        generated_code = generated_code.strip().lstrip('`').lstrip()

        if 'from typing import' not in generated_code:
            generated_code = 'from typing import List\n' + generated_code

        with open("temp_solution.py", "w") as f:
            f.write(generated_code)

        spec = importlib.util.spec_from_file_location("temp_solution", "temp_solution.py")
        temp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(temp_module)
        solution_func = getattr(temp_module, problem['entry_point'])

        test_results = []
        detailed_results = []

        assertions = [line.strip() for line in problem['test'].splitlines() if line.strip().startswith("assert")]

        for i, assertion in enumerate(assertions):
            try:
                exec_globals = {'candidate': solution_func}
                exec(assertion.strip(), exec_globals)
                test_results.append(True)
                detailed_results.append((i, "PASSED", assertion.strip()))
            except Exception as e:
                test_results.append(False)
                detailed_results.append((i, f"FAILED - {str(e)}", assertion.strip()))

        os.remove("temp_solution.py")

        return {
            'passed_tests': sum(test_results),
            'total_tests': len(test_results),
            'success_rate': sum(test_results) / len(test_results),
            'detailed': detailed_results
        }

    def clean_generated_code(self, code: str) -> str:
        match = re.search(r'```(?:python)?\n([\s\S]+?)```', code)
        if match:
            return match.group(1).strip()

        lines = code.strip().splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                return "\n".join(lines[i:]).strip()

        return code.strip()

    def run_benchmark(self, num_problems: int = 5) -> List[Dict]:
        results = []

        for i, problem in enumerate(self.dataset['test']):
            if i >= num_problems:
                break

            print(f"\n{'='*50}")
            print(f"[{self.model_name}] Processing problem {i+1}/{num_problems}")
            print(f"Function: {problem['entry_point']}")
            print(f"{'='*50}")

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

            generated_code = self.clean_generated_code(str(crew_output))
            evaluation = self.evaluate_solution(problem, generated_code)

            print("\nTest Results:")
            for test_id, status, code in evaluation['detailed']:
                print(f"  [{status}] Test {test_id + 1}: {code}")

            print("\nCanonical Solution (reference):\n" + problem['canonical_solution'].strip())
            print("\nGenerated Code:\n" + generated_code)

            results.append({
                'problem_id': i,
                'task_id': problem['task_id'],
                'function_name': problem['entry_point'],
                'execution_time': end_time - start_time,
                'evaluation': evaluation,
                'generated_code': generated_code
            })

            print(f"Success Rate: {evaluation['success_rate']*100:.2f}%")
            print(f"Execution Time: {end_time - start_time:.2f} seconds")
            print(f"{'='*50}")

        return results

    def save_results(self, results: List[Dict], filename: str = None):
        if filename is None:
            filename = f"benchmark_results_{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {filename}")


def print_comparison_summary(deepseek_results: List[Dict], gemini_results: List[Dict]):
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

    widths = [max(len(str(row[i])) for row in [headers] + metrics) for i in range(3)]

    print("".join(f"{headers[i]:<{widths[i]+4}}" for i in range(3)))
    print("-" * (sum(widths) + 8))

    for row in metrics:
        print("".join(f"{str(row[i]):<{widths[i]+4}}" for i in range(3)))

    print("="*70)


def main():
    load_dotenv()
    num_problems = 5

    deepseek_config = {
        "model": "deepseek/deepseek-chat",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "temperature": 0.1
    }

    gemini_config = {
        "model": "gemini/gemini-2.5-pro-exp-03-25",
        "temperature": 0.1,
        "api_key": os.getenv("GEMINI_API_KEY")
    }

    print("\nRunning Deepseek benchmark...")
    deepseek_benchmark = BenchmarkSystem(deepseek_config, "Deepseek")
    deepseek_results = deepseek_benchmark.run_benchmark(num_problems=num_problems)
    deepseek_benchmark.save_results(deepseek_results)

    print("\nRunning Gemini benchmark...")
    gemini_benchmark = BenchmarkSystem(gemini_config, "Gemini")
    gemini_results = gemini_benchmark.run_benchmark(num_problems=num_problems)
    gemini_benchmark.save_results(gemini_results)

    print_comparison_summary(deepseek_results, gemini_results)

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
