
from pathlib import Path
import sys
import json
import re

import pandas as pd

# Make the project root importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from refund_agent import run_agent


SCENARIOS_PATH = ROOT / "data" / "raw" / "agent_scenarios.csv"
OUTPUT_PATH = ROOT / "outputs" / "agent_runs.jsonl"


def get_refund_amount(steps):
    """
    Extract the refund amount from the agent's tool observations.
    Checks the calculation, issue, and escalation tool results.
    """
    for step in steps:
        observation = step.get("observation", {})

        if isinstance(observation, str):
            try:
                observation = json.loads(observation)
            except json.JSONDecodeError:
                continue

        if isinstance(observation, dict):
            if "refund_inr" in observation:
                return float(observation["refund_inr"])

    return 0.0


def normalize_expected_outcome(value):
    value = str(value).strip().upper()

    if value == "FALLBACK":
        return "FALLBACK_MAX_ITERATIONS"

    return value


def main():
    if not SCENARIOS_PATH.exists():
        raise FileNotFoundError(
            f"Scenario file not found: {SCENARIOS_PATH}"
        )

    scenarios = pd.read_csv(SCENARIOS_PATH)

    required_columns = {
        "scenario_id",
        "message",
        "expected_outcome",
        "expected_refund_inr",
    }

    missing_columns = required_columns - set(scenarios.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in scenario CSV: {missing_columns}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []

    # Start a fresh scenario-run results file.
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:

        for _, scenario in scenarios.iterrows():

            scenario_id = str(scenario["scenario_id"]).strip()
            message = str(scenario["message"]).strip()

            expected_status = normalize_expected_outcome(
                scenario["expected_outcome"]
            )

            expected_refund = float(
                scenario["expected_refund_inr"]
            )

            print("\n" + "=" * 65)
            print(f"Running scenario: {scenario_id}")
            print("=" * 65)

            result = run_agent(
                user_message=message,
                scenario_id=scenario_id,
                max_iterations=6,
                token_limit=6000,
                time_limit_seconds=10,
            )

            actual_status = result["status"]
            actual_refund = get_refund_amount(
                result.get("steps", [])
            )

            status_pass = actual_status == expected_status
            refund_pass = abs(
                actual_refund - expected_refund
            ) <= 0.01

            passed = status_pass and refund_pass

            run_record = {
                "scenario_id": scenario_id,
                "status": actual_status,
                "final_answer": result.get("final_answer", ""),
                "llm_calls": result.get("llm_calls", 0),
                "tokens_used": result.get("tokens_used", 0),
                "steps": result.get("steps", []),
            }

            output_file.write(
                json.dumps(run_record, ensure_ascii=False) + "\n"
            )
            output_file.flush()

            results.append({
                "scenario_id": scenario_id,
                "expected_status": expected_status,
                "actual_status": actual_status,
                "expected_refund_inr": expected_refund,
                "actual_refund_inr": actual_refund,
                "llm_calls": result.get("llm_calls", 0),
                "tokens_used": result.get("tokens_used", 0),
                "status_pass": status_pass,
                "refund_pass": refund_pass,
                "overall_pass": passed,
            })

            print(f"Expected status : {expected_status}")
            print(f"Actual status   : {actual_status}")
            print(f"Expected refund : INR {expected_refund:.2f}")
            print(f"Actual refund   : INR {actual_refund:.2f}")
            print(f"LLM calls       : {result.get('llm_calls', 0)}")
            print(f"Tokens used     : {result.get('tokens_used', 0)}")
            print(f"Final answer    : {result.get('final_answer', '')}")
            print(f"Validation      : {'PASS' if passed else 'FAIL'}")

    results_df = pd.DataFrame(results)

    print("\n\n" + "=" * 75)
    print("PART D - SEVEN SCENARIO RESULTS")
    print("=" * 75)

    print(
        results_df[
            [
                "scenario_id",
                "expected_status",
                "actual_status",
                "expected_refund_inr",
                "actual_refund_inr",
                "llm_calls",
                "tokens_used",
                "overall_pass",
            ]
        ].to_string(index=False)
    )

    passed_count = int(results_df["overall_pass"].sum())

    print("\nScenarios passed:", passed_count, "/", len(results_df))
    print("Scenario log saved to:", OUTPUT_PATH)

    if passed_count == len(results_df):
        print("\nPART D STEP 5 COMPLETED.")
    else:
        print("\nSome scenarios did not pass. Review the output above.")


if __name__ == "__main__":
    main()
