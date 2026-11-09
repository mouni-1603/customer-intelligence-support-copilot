
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm
from refund_agent import run_agent


def load_s1_message():
    scenarios_path = ROOT / "data" / "raw" / "agent_scenarios.csv"

    import pandas as pd

    scenarios = pd.read_csv(scenarios_path)
    s1 = scenarios.loc[scenarios["scenario_id"] == "S1"]

    if s1.empty:
        raise ValueError("Scenario S1 was not found.")

    return str(s1.iloc[0]["message"])


def main():
    s1_message = load_s1_message()

    print("=" * 65)
    print("PART D - AGENT SAFETY TESTS")
    print("=" * 65)

    # ---------------------------------------------------------
    # TEST 1: TOKEN LIMIT
    # ---------------------------------------------------------
    print("\nTEST 1: TOKEN LIMIT")
    print("-" * 65)

    mock_llm.reset()

    token_result = run_agent(
        user_message=s1_message,
        scenario_id="S1_TOKEN_LIMIT_TEST",
        max_iterations=6,
        token_limit=200,
        time_limit_seconds=10,
    )

    print("Status:", token_result["status"])
    print("LLM calls:", token_result["llm_calls"])
    print("Tokens used:", token_result["tokens_used"])
    print("Tools executed:", [
        step["tool_name"] for step in token_result["steps"]
    ])
    print("Final answer:", token_result["final_answer"])

    token_pass = (
        token_result["status"] == "FALLBACK_TOKEN_LIMIT"
        and token_result["tokens_used"] > 200
        and token_result["llm_calls"] == 1
        and len(token_result["steps"]) == 0
    )

    print("Token limit validation:", "PASS" if token_pass else "FAIL")

    # ---------------------------------------------------------
    # TEST 2: TIME LIMIT
    # ---------------------------------------------------------
    print("\nTEST 2: TIME LIMIT")
    print("-" * 65)

    mock_llm.reset()

    time_result = run_agent(
        user_message=s1_message,
        scenario_id="S1_TIME_LIMIT_TEST",
        max_iterations=6,
        token_limit=6000,
        time_limit_seconds=0,
    )

    print("Status:", time_result["status"])
    print("LLM calls:", time_result["llm_calls"])
    print("Tokens used:", time_result["tokens_used"])
    print("Tools executed:", [
        step["tool_name"] for step in time_result["steps"]
    ])
    print("Final answer:", time_result["final_answer"])

    time_pass = (
        time_result["status"] == "FALLBACK_TIMEOUT"
        and time_result["llm_calls"] == 0
        and time_result["tokens_used"] == 0
        and len(time_result["steps"]) == 0
    )

    print("Time limit validation:", "PASS" if time_pass else "FAIL")

    # ---------------------------------------------------------
    # TEST 3: MAXIMUM ITERATIONS
    # ---------------------------------------------------------
    print("\nTEST 3: MAXIMUM ITERATIONS")
    print("-" * 65)

    scenarios_path = ROOT / "data" / "raw" / "agent_scenarios.csv"
    import pandas as pd

    scenarios = pd.read_csv(scenarios_path)
    s7 = scenarios.loc[scenarios["scenario_id"] == "S7"]

    if s7.empty:
        raise ValueError("Scenario S7 was not found.")

    mock_llm.reset()

    iteration_result = run_agent(
        user_message=str(s7.iloc[0]["message"]),
        scenario_id="S7_ITERATION_LIMIT_TEST",
        max_iterations=6,
        token_limit=6000,
        time_limit_seconds=10,
    )

    print("Status:", iteration_result["status"])
    print("LLM calls:", iteration_result["llm_calls"])
    print("Tokens used:", iteration_result["tokens_used"])
    print("Tools executed:", [
        step["tool_name"] for step in iteration_result["steps"]
    ])
    print("Final answer:", iteration_result["final_answer"])

    iteration_pass = (
        iteration_result["status"] == "FALLBACK_MAX_ITERATIONS"
        and iteration_result["llm_calls"] == 6
    )

    print(
        "Maximum iteration validation:",
        "PASS" if iteration_pass else "FAIL"
    )

    # ---------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------
    all_passed = token_pass and time_pass and iteration_pass

    print("\n" + "=" * 65)
    print("SAFETY TEST SUMMARY")
    print("=" * 65)
    print("Token limit:", "PASS" if token_pass else "FAIL")
    print("Time limit:", "PASS" if time_pass else "FAIL")
    print("Maximum iterations:", "PASS" if iteration_pass else "FAIL")

    if all_passed:
        print("\nALL SAFETY TESTS PASSED.")
        print("PART D STEP 6 COMPLETED.")
    else:
        print("\nONE OR MORE SAFETY TESTS FAILED.")
        print("Review the results before proceeding.")


if __name__ == "__main__":
    main()
