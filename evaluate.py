"""
This file will be used to evaluate the performance of your agent.
Make sure to set the API key in the `my_agent/.env` file. See `README.md` for more details.
"""

import argparse
import datetime
import json
import os
import pathlib
import shutil
import time

import dotenv
import pydantic
import pyfiglet
from colorama import Fore, Style, init
from google import genai

from utils import server

# Initialize colorama for cross-platform color support
init(autoreset=True)


def _term_width() -> int:
    """Return current terminal column width, with an 80-column fallback."""
    return shutil.get_terminal_size(fallback=(80, 24)).columns


dotenv.load_dotenv(dotenv_path=pathlib.Path("my_agent/.env"))


class JudgeResponse(pydantic.BaseModel):
    """Pydantic model for LLM judge response."""

    is_correct: bool


# Initialize clients for LLM judge (only needed if string matching fails)
# Primary: workshop key, Fallback: personal key
_api_keys = []
_primary_key = os.getenv("GOOGLE_API_KEY")
_fallback_key = os.getenv("GOOGLE_API_KEY_FALLBACK")
if _primary_key:
    _api_keys.append(_primary_key)
if _fallback_key and _fallback_key != _primary_key:
    _api_keys.append(_fallback_key)
client = genai.Client(api_key=_api_keys[0]) if _api_keys else None
if not _api_keys:
    print("Warning: GOOGLE_API_KEY not set. LLM judge will not be available.")


DATASET_PATH = "benchmark/questions.json"
ATTACHMENTS_FOLDER_PATH = "benchmark/attachments"


def print_banner():
    """Print the ML6 banner."""
    banner = pyfiglet.figlet_format("ML6", font="slant")
    width = _term_width()
    # Orange color using ANSI 256-color code
    orange = "\033[38;5;208m"
    print(f"\n{orange}{Style.BRIGHT}{banner}{Style.RESET_ALL}")
    print(f"{orange}{'=' * width}{Style.RESET_ALL}")
    print(
        f"{orange}{Style.BRIGHT}AISO Workshop - Agent Evaluation System{Style.RESET_ALL}"
    )
    print(f"{orange}{'=' * width}{Style.RESET_ALL}\n")


def _load_dataset():
    """Load the train dataset."""
    dataset_path = DATASET_PATH

    try:
        with open(dataset_path, "r") as f:
            data = json.load(f)
            # Handle both array format and dict with "dataset" key
            if isinstance(data, dict) and "dataset" in data:
                return data["dataset"]
            return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in dataset file: {e}")


def string_match(response: str, expected_answer: str) -> bool:
    """
    Check if response matches expected answer using string comparison.
    Handles both exact matches and partial matches.
    """
    resp = response.strip().lower()
    expected = expected_answer.strip().lower()

    # Strip <think>...</think> blocks from response
    import re
    resp_clean = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()

    # Exact match
    if resp_clean == expected:
        return True
    # Exact match on original
    if resp == expected:
        return True
    # Expected answer appears as a standalone token in the response
    if re.search(r"(?<!\w)" + re.escape(expected) + r"(?!\w)", resp_clean):
        return True

    return False


def llm_judge(response: str, expected_answer: str, question: str) -> bool:
    """
    Use LLM as a judge to determine if the response is correct.
    Returns a boolean indicating if the response is correct.
    """
    if client is None:
        raise ValueError("GOOGLE_API_KEY not set")
    if response is None or response.strip() == "":
        return False
    prompt = f"""
    You are an evaluation judge.
    Determine if the agent's response is semantically completely equivalent to the expected answer.

Question: {question}

Expected Answer: {expected_answer}

Agent's Response: {response}

Evaluate whether the agent's response is semantically equivalent to the expected answer, even if worded differently.
Be strict but fair - minor variations in wording are acceptable if the core answer is correct.
"""

    for key in _api_keys:
        judge_client = genai.Client(api_key=key)
        try:
            llm_response = judge_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": JudgeResponse,
                },
            )

            parsed = llm_response.parsed
            if isinstance(parsed, JudgeResponse):
                return parsed.is_correct
            if isinstance(parsed, dict) and "is_correct" in parsed:
                return bool(parsed["is_correct"])
            raise ValueError(f"Unexpected LLM judge payload: {type(parsed).__name__}")
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"Rate limited on key ...{key[-6:]}, trying next key...")
                continue
            print(f"Error in LLM judge: {e}")
            raise e
    raise ValueError("All API keys exhausted (rate limited)")


def evaluate_single_question(question_data: dict, question_idx: int) -> dict:
    """
    Evaluate a single question.

    Args:
        question_data: Dict containing question, answer, and optional file_name
        question_idx: Index of the question in the dataset

    Returns:
        Dict with evaluation results
    """
    # Extract question and answer based on dataset format
    if "Question" in question_data:  # verbose format
        question = question_data["Question"]
        expected_answer = question_data["Final answer"]
        file_name = question_data.get("file_name", "")
    else:  # simple format
        question = question_data["question"]
        expected_answer = question_data["answer"]
        file_name = question_data.get("file_name", "")

    # Prepare file paths if files are provided
    file_paths = None
    if file_name:
        # Handle comma-separated file names
        files = [f.strip() for f in file_name.split(",") if f.strip()]
        if files:
            file_paths = [f"{ATTACHMENTS_FOLDER_PATH}/{f}" for f in files]

    width = _term_width()
    print(f"\n{Fore.CYAN}{'=' * width}")
    print(f"{Fore.CYAN}{Style.BRIGHT}Question {question_idx + 1}")
    print(f"{Fore.CYAN}{'=' * width}{Style.RESET_ALL}")
    question_display = f"{question[:100]}..." if len(question) > 100 else question
    print(f"{Fore.BLUE}{Style.BRIGHT}Question:{Style.RESET_ALL} {question_display}")
    if file_paths:
        print(f"{Fore.MAGENTA}Files:{Style.RESET_ALL} {file_paths}")

    # Run the agent (using USER_ID env var if set, otherwise default "dev_user")
    user_id = os.getenv("USER_ID", "dev_user")
    tool_calls: list[str] = []
    try:
        start_time = time.time()
        agent_details = server.run_agent(question, file_paths, user_id=user_id)
        end_time = time.time()
        response_time = end_time - start_time
        agent_response = agent_details["response_text"]
        tool_calls = agent_details.get("tool_calls", [])

        print(f"\n{Fore.WHITE}Agent Response:{Style.RESET_ALL} {agent_response}")
        if tool_calls:
            print(f"{Fore.CYAN}Tool Calls:{Style.RESET_ALL} {' -> '.join(tool_calls)}")
        else:
            print(f"{Fore.CYAN}Tool Calls:{Style.RESET_ALL} (none)")
        print(f"{Fore.YELLOW}Expected Answer:{Style.RESET_ALL} {expected_answer}")
        print(f"{Fore.MAGENTA}Response Time:{Style.RESET_ALL} {response_time:.2f}s")
    except Exception as error:
        end_time = time.time()
        response_time = end_time - start_time
        error_message = str(error)
        print(f"{Fore.RED}Error running agent: {error_message}{Style.RESET_ALL}")
        return {
            "question_idx": question_idx,
            "question": question,
            "expected_answer": expected_answer,
            "agent_response": "",
            "correct": False,
            "method": "error",
            "error": error_message,
            "tool_calls": tool_calls,
            "response_time": response_time,
        }

    # First try string matching
    string_matches = string_match(agent_response, expected_answer)

    if string_matches:
        print(f"{Fore.GREEN}{Style.BRIGHT}✓ Correct (string match){Style.RESET_ALL}")
        return {
            "question_idx": question_idx,
            "question": question,
            "expected_answer": expected_answer,
            "agent_response": agent_response,
            "correct": True,
            "method": "string_match",
            "tool_calls": tool_calls,
            "response_time": response_time,
        }

    # Fall back to LLM judge
    print(f"\n{Fore.YELLOW}String match failed, using LLM judge...{Style.RESET_ALL}")
    try:
        is_correct = llm_judge(agent_response, expected_answer, question)
        method = "llm_judge"
    except Exception as error:
        print(f"{Fore.RED}LLM judge failed: {error}{Style.RESET_ALL}")
        is_correct = False
        method = "llm_judge_error"

    if is_correct:
        print(f"{Fore.GREEN}{Style.BRIGHT}✓ Correct (LLM judge){Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}{Style.BRIGHT}✗ Incorrect{Style.RESET_ALL}")

    return {
        "question_idx": question_idx,
        "question": question,
        "expected_answer": expected_answer,
        "agent_response": agent_response,
        "correct": is_correct,
        "method": method,
        "tool_calls": tool_calls,
        "response_time": response_time,
    }


def evaluate_all(dataset_path=None, output_file=None) -> dict:
    """
    Evaluate all questions in the dataset.

    Returns:
        Dict with aggregated results
    """
    # Print the banner
    print_banner()

    dataset = _load_dataset()

    results = []
    correct_count = 0
    total_count = len(dataset)

    print(
        f"{Fore.CYAN}{Style.BRIGHT}Starting evaluation of {total_count} questions...{Style.RESET_ALL}"
    )
    print(f"{Fore.CYAN}{'=' * _term_width()}{Style.RESET_ALL}")

    for idx, question_data in enumerate(dataset):
        result = evaluate_single_question(question_data, idx)
        results.append(result)

        if result["correct"]:
            correct_count += 1

    # Calculate accuracy
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

    # Calculate timing statistics
    response_times = [r["response_time"] for r in results]
    avg_response_time = (
        sum(response_times) / len(response_times) if response_times else 0
    )

    # Calculate metrics for correct answers only
    correct_response_times = [r["response_time"] for r in results if r["correct"]]
    avg_correct_response_time = (
        sum(correct_response_times) / len(correct_response_times)
        if correct_response_times
        else 0
    )

    # Prepare summary
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_questions": total_count,
        "correct": correct_count,
        "incorrect": total_count - correct_count,
        "accuracy": round(accuracy, 2),
        "timing": {
            "average_response_time": round(avg_response_time, 2),
            "average_correct_response_time": round(avg_correct_response_time, 2),
        },
        "results": results,
    }

    # Print summary
    width = _term_width()
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * width}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * width}{Style.RESET_ALL}")
    print(f"\n{Fore.WHITE}{Style.BRIGHT}Correctness Metrics:{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Total Questions:{Style.RESET_ALL} {total_count}")
    print(f"{Fore.GREEN}Correct:{Style.RESET_ALL} {correct_count}")
    print(f"{Fore.RED}Incorrect:{Style.RESET_ALL} {total_count - correct_count}")
    print(f"{Fore.CYAN}{Style.BRIGHT}Accuracy:{Style.RESET_ALL} {accuracy:.2f}%")
    print(f"\n{Fore.WHITE}{Style.BRIGHT}Timing Metrics:{Style.RESET_ALL}")
    print(
        f"{Fore.MAGENTA}Average Response Time (All):{Style.RESET_ALL} {avg_response_time:.2f}s"
    )
    print(
        f"{Fore.GREEN}Average Response Time (Correct Only):{Style.RESET_ALL} {avg_correct_response_time:.2f}s"
    )
    print(f"{Fore.CYAN}{'=' * _term_width()}{Style.RESET_ALL}")

    # Save results to file
    if output_file is None:
        os.makedirs("results", exist_ok=True)
        output_file = f"results/evaluation_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    try:
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n{Fore.CYAN}Results saved to:{Style.RESET_ALL} {output_file}")
    except IOError as e:
        raise IOError(f"Failed to write results to {output_file}: {e}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the agent on train dataset")
    parser.add_argument(
        "--question",
        type=int,
        help=(
            "Question number to evaluate (1-based). "
            "If not provided, evaluates all questions."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for results. Default: results/evaluation_results_<timestamp>.json",
    )

    args = parser.parse_args()

    if args.question is not None:
        # Evaluate single question
        print_banner()

        dataset = _load_dataset()
        question_idx = args.question - 1
        if question_idx < 0 or question_idx >= len(dataset):
            raise ValueError(
                f"Question number {args.question} out of range (1-{len(dataset)})"
            )

        result = evaluate_single_question(dataset[question_idx], question_idx)
        if result["correct"]:
            print(f"\n{Fore.GREEN}{Style.BRIGHT}Result: ✓ Correct{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.RED}{Style.BRIGHT}Result: ✗ Incorrect{Style.RESET_ALL}")

        print(
            f"{Fore.MAGENTA}Response Time:{Style.RESET_ALL} {result['response_time']:.2f}s"
        )
    else:
        # Evaluate all questions
        evaluate_all(output_file=args.output)
