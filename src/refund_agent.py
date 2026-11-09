
from pathlib import Path
import sys
import json
import time
import uuid

# ============================================================
# 1. PROJECT IMPORTS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mock_llm
from refund_tools import TOOLS_SPEC, TOOL_MAP


# ============================================================
# 2. CHECKPOINT CONFIGURATION
# ============================================================

STATE_DIR = ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)


class SimulatedCrash(Exception):
    """Raised to demonstrate recovery from an interrupted run."""

    def __init__(self, run_id):
        self.run_id = run_id
        super().__init__(
            f"Simulated crash after tool execution. Run ID: {run_id}"
        )


# ============================================================
# 3. AGENT SYSTEM PROMPT
# ============================================================

AGENT_SYSTEM_PROMPT = """
You are an AGENT responsible for processing customer refund requests.

You must use the available tools to make decisions.

Follow this process:

1. Identify the order ID from the customer's message.
2. Call get_order to retrieve the order details.
3. Call search_policy to retrieve the relevant refund policy.
4. Only completed orders are eligible for refunds.
5. Call calculate_refund to calculate the eligible refund.
6. If the refund amount is zero, deny the refund.
7. If the refund amount is greater than INR 1000,
   call escalate_to_human.
8. If the refund amount is INR 1000 or less,
   call issue_refund.
9. If the order does not exist, return NOT_FOUND.
10. Never issue a refund for a cancelled order.

Use the following final response statuses:

REFUND: refund successfully issued.
ESCALATE: refund sent for human approval.
DENIED: refund request is not eligible.
NOT_FOUND: order does not exist.

Use the tools rather than inventing order details.
"""


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def extract_status(final_answer):

    if not final_answer:
        return "UNKNOWN"

    return final_answer.split(":", 1)[0].strip()


def make_result(
    status,
    final_answer,
    steps,
    llm_calls,
    tokens_used,
    run_id=None
):

    return {
        "run_id": run_id,
        "status": status,
        "final_answer": final_answer,
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used
    }


def checkpoint_path(run_id):
    return STATE_DIR / f"{run_id}.json"


def save_checkpoint(state):

    path = checkpoint_path(state["run_id"])

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            state,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\nCheckpoint saved:", path)


def load_checkpoint(run_id):

    path = checkpoint_path(run_id)

    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for run ID: {run_id}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# 5. AGENT EXECUTION ENGINE
# ============================================================

def continue_agent(state, crash_after_tools=None):

    run_id = state["run_id"]
    messages = state["messages"]
    steps = state["steps"]
    tools_executed = state["tools_executed"]

    llm_calls = state["llm_calls"]
    tokens_used = state["tokens_used"]

    max_iterations = state["max_iterations"]
    token_limit = state["token_limit"]
    time_limit_seconds = state["time_limit_seconds"]

    start_time = time.monotonic()

    # --------------------------------------------------------
    # ITERATION LOOP
    # --------------------------------------------------------

    while llm_calls < max_iterations:

        elapsed = time.monotonic() - start_time

        if elapsed >= time_limit_seconds:

            return make_result(
                "FALLBACK_TIMEOUT",
                "FALLBACK_TIMEOUT: Processing time limit reached. "
                "Please transfer this request to a human agent.",
                steps,
                llm_calls,
                tokens_used,
                run_id
            )

        # ----------------------------------------------------
        # CALL MOCK LLM
        # ----------------------------------------------------

        response = mock_llm.chat(
            messages,
            tools=TOOLS_SPEC
        )

        llm_calls += 1

        call_tokens = response.usage.total_tokens
        tokens_used += call_tokens

        print(
            f"\nLLM call {llm_calls}: "
            f"{call_tokens} tokens; total={tokens_used}"
        )

        # ----------------------------------------------------
        # TOKEN LIMIT CHECK
        # ----------------------------------------------------

        if tokens_used > token_limit:

            return make_result(
                "FALLBACK_TOKEN_LIMIT",
                "FALLBACK_TOKEN_LIMIT: Token budget exceeded. "
                "Please transfer this request to a human agent.",
                steps,
                llm_calls,
                tokens_used,
                run_id
            )

        message = response.choices[0].message
        tool_calls = message.tool_calls

        # ----------------------------------------------------
        # FINAL ANSWER
        # ----------------------------------------------------

        if not tool_calls:

            final_answer = message.content or (
                "FALLBACK: Agent returned an empty response."
            )

            status = extract_status(final_answer)

            return make_result(
                status,
                final_answer,
                steps,
                llm_calls,
                tokens_used,
                run_id
            )

        # ----------------------------------------------------
        # APPEND ASSISTANT TOOL-CALL MESSAGE
        # ----------------------------------------------------

        assistant_tool_calls = []

        for tool_call in tool_calls:

            assistant_tool_calls.append({
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments
            })

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_tool_calls
        })

        # ----------------------------------------------------
        # EXECUTE TOOLS
        # ----------------------------------------------------

        for tool_call in tool_calls:

            tool_name = tool_call.name
            arguments = tool_call.arguments
            tool_call_id = tool_call.id

            print("\nExecuting tool:", tool_name)
            print("Arguments:", arguments)

            if tool_name not in TOOL_MAP:

                observation = {
                    "error": f"Unknown tool: {tool_name}"
                }

            else:

                try:
                    observation = TOOL_MAP[tool_name](
                        **arguments
                    )

                except Exception as error:

                    observation = {
                        "error": str(error)
                    }

            # ------------------------------------------------
            # RECORD TOOL EXECUTION
            # ------------------------------------------------

            step = {
                "step": len(steps) + 1,
                "tool_name": tool_name,
                "arguments": arguments,
                "observation": observation
            }

            steps.append(step)
            tools_executed.append(tool_name)

            print("Observation:", observation)

            # ------------------------------------------------
            # APPEND TOOL OBSERVATION TO CONVERSATION
            # ------------------------------------------------

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": json.dumps(
                    observation,
                    ensure_ascii=False
                )
            })

            # ------------------------------------------------
            # SAVE CHECKPOINT AFTER EVERY TOOL
            # ------------------------------------------------

            state["messages"] = messages
            state["steps"] = steps
            state["tools_executed"] = tools_executed
            state["llm_calls"] = llm_calls
            state["tokens_used"] = tokens_used

            save_checkpoint(state)

            # ------------------------------------------------
            # SIMULATE CRASH FOR RECOVERY DEMONSTRATION
            # ------------------------------------------------

            if (
                crash_after_tools is not None
                and len(steps) >= crash_after_tools
            ):

                raise SimulatedCrash(run_id)

    # --------------------------------------------------------
    # MAXIMUM ITERATIONS FALLBACK
    # --------------------------------------------------------

    return make_result(
        "FALLBACK_MAX_ITERATIONS",
        "FALLBACK_MAX_ITERATIONS: Maximum agent iterations "
        "reached without a final answer. "
        "Please transfer this request to a human agent.",
        steps,
        llm_calls,
        tokens_used,
        run_id
    )


# ============================================================
# 6. START A NEW AGENT RUN
# ============================================================

def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tools=None
):

    run_id = f"{scenario_id}_{uuid.uuid4().hex[:8]}"

    messages = [
        {
            "role": "system",
            "content": AGENT_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "user_message": user_message,
        "messages": messages,
        "steps": [],
        "tools_executed": [],
        "llm_calls": 0,
        "tokens_used": 0,
        "max_iterations": max_iterations,
        "token_limit": token_limit,
        "time_limit_seconds": time_limit_seconds
    }

    return continue_agent(
        state,
        crash_after_tools=crash_after_tools
    )


# ============================================================
# 7. RESUME AN INTERRUPTED AGENT RUN
# ============================================================

def resume_agent(run_id):

    print("=" * 65)
    print("RESUMING AGENT FROM CHECKPOINT")
    print("=" * 65)

    state = load_checkpoint(run_id)

    print("Run ID:", run_id)
    print("Scenario:", state["scenario_id"])
    print("Previously completed tools:", state["tools_executed"])
    print("Previous LLM calls:", state["llm_calls"])
    print("Previous tokens:", state["tokens_used"])

    result = continue_agent(state)

    return result


# ============================================================
# 8. BASIC AGENT TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("PART D - CHECKPOINT-ENABLED REFUND AGENT TEST")
    print("=" * 65)

    test_message = (
        "My order ORD-100001 arrived. "
        "I would like a refund."
    )

    mock_llm.reset()

    result = run_agent(
        user_message=test_message,
        scenario_id="TEST",
        max_iterations=6,
        token_limit=6000,
        time_limit_seconds=10
    )

    print("\n" + "=" * 65)
    print("AGENT FINAL RESULT")
    print("=" * 65)

    print("Run ID:", result["run_id"])
    print("Status:", result["status"])
    print("Final answer:", result["final_answer"])
    print("Model calls:", result["llm_calls"])
    print("Tokens used:", result["tokens_used"])

    print("\nTOOLS EXECUTED")

    for step in result["steps"]:

        print(
            step["step"],
            step["tool_name"],
            step["arguments"],
            step["observation"]
        )
