
import json
import re
import random
import time
from pathlib import Path
from pathlib import Path
import sys

# Add the project root to Python's import path.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import mock_llm



# ==================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"

TICKETS_FILE = RAW / "tickets.csv"

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3

CATEGORIES = ["billing", "delivery", "technical", "fraud"]


# ==================================================
# 2. PRIVACY REDACTION PATTERNS
# Apply in the exact specified order.
# ==================================================

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")
ORDER_RE = re.compile(r"ORD-\d{6}")


def redact_text(text):

    email_count = 0
    phone_count = 0
    order_count = 0

    text, email_count = EMAIL_RE.subn("[EMAIL]", text)
    text, phone_count = PHONE_RE.subn("[PHONE]", text)
    text, order_count = ORDER_RE.subn("[ORDER_ID]", text)

    counts = {
        "emails": email_count,
        "phones": phone_count,
        "order_ids": order_count
    }

    return text, counts


# ==================================================
# 3. ZERO-SHOT AND FEW-SHOT SYSTEM PROMPTS
# ==================================================

ZERO_SHOT_PROMPT = """
You are a customer support ticket classification assistant.

CLASSIFY each support ticket into exactly one of these
four allowed categories:

- billing
- delivery
- technical
- fraud

Read the ticket carefully and identify the main issue.

Return a JSON-only response in exactly this format:
{"category": "...", "reason": "..."}

Do not include Markdown, explanations outside JSON,
or any additional text.
"""

FEW_SHOT_PROMPT = """
You are a customer support ticket classification assistant.

CLASSIFY each support ticket into exactly one of these
four allowed categories:

- billing
- delivery
- technical
- fraud

Return a JSON-only response in exactly this format:
{"category": "...", "reason": "..."}

EXAMPLES:

Example 1:
Ticket: I was charged twice for my purchase.
Category: billing

Example 2:
Ticket: My account was hacked and I noticed an unauthorized transaction.
Category: fraud

Example 3:
Ticket: My parcel has not arrived and the delivery is late.
Category: delivery

Use the examples as guidance. Classify the new ticket
and return only the required JSON object.
"""


# ==================================================
# 4. LOAD TICKETS AND REDACT ALL TEXT
# ==================================================

def load_and_redact_tickets():

    # keep_default_na=False preserves empty ticket strings.
    tickets = pd.read_csv(
        TICKETS_FILE,
        keep_default_na=False
    )

    total_counts = {
        "emails": 0,
        "phones": 0,
        "order_ids": 0
    }

    redacted_texts = []

    for original_text in tickets["text"]:

        redacted, counts = redact_text(original_text)

        redacted_texts.append(redacted)

        for key in total_counts:
            total_counts[key] += counts[key]

        # Verify that no PII remains.
        assert not EMAIL_RE.search(redacted)
        assert not PHONE_RE.search(redacted)
        assert not ORDER_RE.search(redacted)

    tickets["redacted_text"] = redacted_texts

    print("=" * 65)
    print("PII REDACTION SUMMARY")
    print("=" * 65)

    print("Total tickets:", len(tickets))
    print("Emails redacted:", total_counts["emails"])
    print("Phone numbers redacted:", total_counts["phones"])
    print("Order IDs redacted:", total_counts["order_ids"])

    total_substitutions = sum(total_counts.values())

    print("Total substitutions:", total_substitutions)
    print("Remaining emails: 0")
    print("Remaining phone numbers: 0")
    print("Remaining order IDs: 0")

    assert total_counts == {
        "emails": 55,
        "phones": 91,
        "order_ids": 132
    }

    return tickets, total_counts


# ==================================================
# 5. RESPONSE VALIDATION
# ==================================================

def validate_response(response_text):

    # First attempt: parse JSON.
    try:
        result = json.loads(response_text)

        # Valid JSON with an invalid category must be rejected.
        if (
            isinstance(result, dict)
            and result.get("category") in CATEGORIES
        ):
            return (
                "classified_json",
                result["category"]
            )

        return "flagged_invalid_output", ""

    except (json.JSONDecodeError, TypeError):

        # Second attempt: extract lowercase words.
        words = re.findall(
            r"[a-z]+",
            response_text.lower()
        )

        matched_categories = set(words).intersection(
            CATEGORIES
        )

        if len(matched_categories) == 1:

            category = matched_categories.pop()

            return (
                "classified_keyword",
                category
            )

        return "flagged_invalid_output", ""


# ==================================================
# 6. RETRY WRAPPER
# ==================================================

def call_with_retry(messages, ticket_id, rng):

    for retry_number in range(MAX_RETRIES + 1):

        try:

            response = mock_llm.chat(messages)

            return response

        except mock_llm.TransientLLMError:

            if retry_number >= MAX_RETRIES:
                raise RuntimeError(
                    f"Ticket {ticket_id} failed after "
                    f"{MAX_RETRIES} retries."
                )

            wait = (
                BASE_DELAY
                * (2 ** retry_number)
                * rng.uniform(0.9, 1.15)
            )

            print(
                f"RETRY | Ticket: {ticket_id} | "
                f"Retry number: {retry_number} | "
                f"Wait: {wait:.5f} seconds"
            )

            time.sleep(wait)


# ==================================================
# 7. RUN THE FULL CLASSIFICATION PIPELINE
# ==================================================


def run_pipeline(tickets, system_prompt, run_name, token_budget=None):

    # Reset mock immediately before each independent full run.
    mock_llm.reset()

    rng = random.Random(42)

    results = []
    total_tokens = 0
    model_calls = 0

    empty_count = 0
    long_count = 0
    budget_exhausted_count = 0

    for _, ticket in tickets.iterrows():

        ticket_id = ticket["ticket_id"]
        customer_id = ticket["customer_id"]

        original_text = ticket["text"]
        redacted_text = ticket["redacted_text"]
        true_category = ticket["true_category"]

        # Input guardrail 1: empty ticket
        if not original_text.strip():

            empty_count += 1

            results.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "status": "skipped_empty",
                "category": "",
                "true_category": true_category,
                "tokens_used": 0
            })

            continue

        # Input guardrail 2: excessively long ticket
        estimated_tokens = mock_llm.est_tokens(original_text)

        if estimated_tokens > MAX_INPUT_TOKENS:

            long_count += 1

            results.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "status": "flagged_long",
                "category": "",
                "true_category": true_category,
                "tokens_used": 0
            })

            continue

        # Privacy assertions before model call
        assert not EMAIL_RE.search(redacted_text)
        assert not PHONE_RE.search(redacted_text)
        assert not ORDER_RE.search(redacted_text)

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": redacted_text
            }
        ]

        assert messages[1]["content"] == redacted_text

        # Token budget guardrail
        if token_budget is not None and total_tokens >= token_budget:

            budget_exhausted_count += 1

            results.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "status": "budget_exhausted",
                "category": "",
                "true_category": true_category,
                "tokens_used": 0
            })

            continue

        # Model call with retry
        response = call_with_retry(messages, ticket_id, rng)

        model_calls += 1

        response_tokens = response.usage.total_tokens
        total_tokens += response_tokens

        response_text = response.choices[0].message.content

        # Validate model response
        status, category = validate_response(response_text)

        results.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "true_category": true_category,
            "tokens_used": response_tokens
        })

    results_df = pd.DataFrame(results)

    classified = results_df[
        results_df["status"].isin(
            ["classified_json", "classified_keyword"]
        )
    ]

    accuracy = (
        (classified["category"] == classified["true_category"]).mean()
        if len(classified) > 0
        else 0
    )

    # Print run summary
    print("\n" + "=" * 65)
    print(run_name.upper() + " RUN SUMMARY")
    print("=" * 65)

    print("Total tickets:", len(results_df))
    print("Empty tickets skipped:", empty_count)
    print("Long tickets flagged:", long_count)
    print("Budget exhausted:", budget_exhausted_count)
    print("Model calls completed:", model_calls)

    print("\nStatus counts:")
    print(results_df["status"].value_counts().to_string())

    print("\nTickets classified:", len(classified))
    print("Classification accuracy:", round(accuracy * 100, 2), "%")
    print("Total tokens:", total_tokens)

    print(
        "Tokens per classified ticket:",
        round(total_tokens / len(classified), 2)
        if len(classified) else 0
    )

    # Zero-shot confusion table
    if run_name == "zero_shot":

        confusion = pd.crosstab(
            classified["true_category"],
            classified["category"],
            rownames=["Actual"],
            colnames=["Predicted"]
        )

        confusion = confusion.reindex(
            index=CATEGORIES,
            columns=CATEGORIES,
            fill_value=0
        )

        print("\nZERO-SHOT CONFUSION TABLE")
        print(confusion.to_string())

    return results_df, {
        "tickets_classified": len(classified),
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "tokens_per_classified_ticket": (
            total_tokens / len(classified)
            if len(classified) else 0
        ),
        "model_calls": model_calls,
        "budget_exhausted": budget_exhausted_count
    }


# ==================================================
# 8. MAIN EXECUTION
# ==================================================


def main():

    OUTPUTS.mkdir(
        parents=True,
        exist_ok=True
    )

    tickets, redaction_counts = load_and_redact_tickets()

    # --------------------------------------------------
    # 1. ZERO-SHOT RUN WITH TOKEN BUDGET
    # --------------------------------------------------

    zero_budget_results, zero_budget_summary = run_pipeline(
        tickets,
        ZERO_SHOT_PROMPT,
        "zero_shot_budget",
        token_budget=6000
    )

    budget_file = OUTPUTS / "triage_budget_demo.csv"

    zero_budget_results.to_csv(
        budget_file,
        index=False
    )

    print("\nBudget demonstration saved to:", budget_file)

    # --------------------------------------------------
    # 2. FULL FEW-SHOT RUN FOR FINAL TICKET ROUTING
    # --------------------------------------------------

    few_results, few_summary = run_pipeline(
        tickets,
        FEW_SHOT_PROMPT,
        "few_shot_final",
        token_budget=100000
    )

    # --------------------------------------------------
    # 3. LOAD CUSTOMER SEGMENT INFORMATION
    # --------------------------------------------------

    segments_file = OUTPUTS / "customer_segments.csv"

    if not segments_file.exists():
        raise FileNotFoundError(
            "customer_segments.csv not found. "
            "Complete Part B customer segmentation first."
        )

    segments = pd.read_csv(segments_file)

    required_columns = [
        "customer_id",
        "segment_name",
        "is_premium"
    ]

    for column in required_columns:
        if column not in segments.columns:
            raise ValueError(
                f"Required column missing from customer_segments.csv: {column}"
            )

    # Convert premium indicator into a consistent boolean.
    segments["is_premium"] = (
        segments["is_premium"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    # Build customer lookup.
    segment_lookup = segments.set_index("customer_id").to_dict("index")

    # --------------------------------------------------
    # 4. APPLY FINAL ROUTING RULES
    # --------------------------------------------------

    final_rows = []

    for _, result in few_results.iterrows():

        customer_id = result["customer_id"]
        status = result["status"]
        category = result["category"]

        customer_segment = segment_lookup.get(customer_id, {})

        segment_name = customer_segment.get("segment_name", "")
        is_premium = customer_segment.get("is_premium", False)

        priority = "NORMAL"
        route = "support_queue"

        # Human review for tickets that cannot be safely classified.
        if status in [
            "skipped_empty",
            "flagged_long",
            "flagged_invalid_output",
            "budget_exhausted"
        ]:

            route = "human_review"
            priority = "NORMAL"

        # Fraud tickets go to the security team.
        elif category == "fraud":

            route = "security_team"
            priority = "HIGH"

        # Premium customers go to the premium desk.
        elif is_premium:

            route = "premium_desk"
            priority = "HIGH"

        # All other classified tickets go to the normal support queue.
        else:

            route = "support_queue"
            priority = "NORMAL"

        final_rows.append({
            "ticket_id": result["ticket_id"],
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": result["tokens_used"]
        })

    final_results = pd.DataFrame(final_rows)

    # --------------------------------------------------
    # 5. SAVE FINAL TRIAGE RESULTS
    # --------------------------------------------------

    final_file = OUTPUTS / "triage_results.csv"

    final_results.to_csv(
        final_file,
        index=False
    )

    print("\nFinal triage results saved to:", final_file)

    # --------------------------------------------------
    # 6. VALIDATION SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("PART C FINAL VALIDATION")
    print("=" * 65)

    print("Final ticket rows:", len(final_results))

    print("\nFinal status counts:")
    print(final_results["status"].value_counts().to_string())

    print("\nRouting counts:")
    print(final_results["route"].value_counts().to_string())

    print("\nPriority counts:")
    print(final_results["priority"].value_counts().to_string())

    print("\nBudget demonstration:")
    print("Token budget: 6000")
    print("Actual tokens used:", zero_budget_summary["total_tokens"])
    print("Budget exhausted tickets:", zero_budget_summary["budget_exhausted"])

    print("\nFinal few-shot run:")
    print("Accuracy:", round(few_summary["accuracy"] * 100, 2), "%")
    print("Total tokens:", few_summary["total_tokens"])

    # Basic acceptance checks.
    assert len(final_results) == 164
    assert len(zero_budget_results) == 164

    assert not (
        final_results["route"].isna().any()
    )

    assert set(final_results["route"]).issubset({
        "human_review",
        "security_team",
        "premium_desk",
        "support_queue"
    })

    assert zero_budget_summary["budget_exhausted"] > 0, (
        "The 6000-token budget did not exhaust. "
        "Try a lower budget such as 5000."
    )

    print("\nALL PART C OUTPUT VALIDATIONS PASSED")


if __name__ == "__main__":
    main()