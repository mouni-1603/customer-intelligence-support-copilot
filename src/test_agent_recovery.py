
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mock_llm
from refund_agent import (
    run_agent,
    resume_agent,
    SimulatedCrash,
    load_checkpoint,
    checkpoint_path,
)


def main():

    print("=" * 70)
    print("PART D - CHECKPOINT AND RECOVERY DEMONSTRATION")
    print("=" * 70)

    user_message = (
        "My order ORD-100048 arrived late. "
        "I would like a refund."
    )

    # --------------------------------------------------------
    # PHASE 1: START THE AGENT AND SIMULATE A CRASH
    # --------------------------------------------------------

    print("\nPHASE 1: START AGENT")
    print("-" * 70)

    mock_llm.reset()

    run_id = None

    try:

        run_agent(
            user_message=user_message,
            scenario_id="S1_RECOVERY",
            max_iterations=6,
            token_limit=6000,
            time_limit_seconds=10,
            crash_after_tools=2,
        )

    except SimulatedCrash as error:

        run_id = error.run_id

        print("\nSIMULATED CRASH SUCCESSFUL")
        print("Run ID:", run_id)

    if run_id is None:
        raise RuntimeError(
            "The simulated crash did not occur."
        )

    # --------------------------------------------------------
    # PHASE 2: VERIFY THE SAVED CHECKPOINT
    # --------------------------------------------------------

    print("\nPHASE 2: VERIFY CHECKPOINT")
    print("-" * 70)

    checkpoint = load_checkpoint(run_id)

    print("Checkpoint file:", checkpoint_path(run_id))
    print("Scenario ID:", checkpoint["scenario_id"])
    print("LLM calls saved:", checkpoint["llm_calls"])
    print("Tokens saved:", checkpoint["tokens_used"])
    print("Tools already executed:", checkpoint["tools_executed"])

    expected_tools_before_resume = [
        "get_order",
        "search_policy",
    ]

    checkpoint_pass = (
        checkpoint["tools_executed"]
        == expected_tools_before_resume
        and len(checkpoint["steps"]) == 2
        and checkpoint["llm_calls"] == 2
    )

    print(
        "Checkpoint validation:",
        "PASS" if checkpoint_pass else "FAIL"
    )

    if not checkpoint_pass:
        raise AssertionError(
            "Checkpoint does not contain the expected progress."
        )

    # --------------------------------------------------------
    # PHASE 3: RESUME THE AGENT
    # --------------------------------------------------------

    print("\nPHASE 3: RESUME AGENT")
    print("-" * 70)

    result = resume_agent(run_id)

    print("\nRESUMED AGENT RESULT")
    print("-" * 70)

    print("Run ID:", result["run_id"])
    print("Status:", result["status"])
    print("Final answer:", result["final_answer"])
    print("Total LLM calls:", result["llm_calls"])
    print("Total tokens:", result["tokens_used"])

    print("\nALL TOOLS EXECUTED IN ORDER")

    for step in result["steps"]:
        print(
            step["step"],
            step["tool_name"],
            step["arguments"],
            step["observation"],
        )

    # --------------------------------------------------------
    # PHASE 4: VALIDATE NO TOOL WAS REPEATED
    # --------------------------------------------------------

    expected_all_tools = [
        "get_order",
        "search_policy",
        "calculate_refund",
        "issue_refund",
    ]

    actual_tools = [
        step["tool_name"]
        for step in result["steps"]
    ]

    refund_amounts = [
        step["observation"].get("refund_inr")
        for step in result["steps"]
        if isinstance(step["observation"], dict)
        and "refund_inr" in step["observation"]
    ]

    final_refund = refund_amounts[-1] if refund_amounts else 0

    tools_pass = actual_tools == expected_all_tools

    refund_pass = (
        result["status"] == "REFUND"
        and abs(float(final_refund) - 128.25) <= 0.01
    )

    recovery_pass = (
        tools_pass
        and refund_pass
        and result["llm_calls"] == 5
    )

    print("\n" + "=" * 70)
    print("RECOVERY VALIDATION SUMMARY")
    print("=" * 70)

    print("Checkpoint saved correctly:", "PASS" if checkpoint_pass else "FAIL")
    print("Tools executed in correct order:", "PASS" if tools_pass else "FAIL")
    print("No repeated tools:", "PASS" if tools_pass else "FAIL")
    print("Final refund INR 128.25:", "PASS" if refund_pass else "FAIL")
    print("Total model calls = 5:", "PASS" if result["llm_calls"] == 5 else "FAIL")

    if recovery_pass:
        print("\nALL RECOVERY TESTS PASSED.")
        print("PART D CHECKPOINT AND RESUME COMPLETED.")
    else:
        print("\nRECOVERY VALIDATION FAILED.")


if __name__ == "__main__":
    main()
