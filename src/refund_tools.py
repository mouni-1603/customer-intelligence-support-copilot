
from pathlib import Path
import sys
import sqlite3
import json
from datetime import datetime, timezone

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATABASE_PATH = ROOT / "data" / "instamart.db"
LEDGER_PATH = ROOT / "outputs" / "refund_ledger.jsonl"

from policy_rag import retrieve


# ============================================================
# 2. TOOL 1: GET ORDER
# ============================================================

def get_order(order_id):

    """
    Retrieve order details from the SQLite database.
    """

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                order_id,
                status,
                order_amount,
                delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,)
        )

        row = cursor.fetchone()

        if row is None:
            return {"error": "order not found"}

        return {
            "order_id": row["order_id"],
            "status": row["status"],
            "order_amount": float(row["order_amount"]),
            "delivery_minutes": (
                float(row["delivery_minutes"])
                if row["delivery_minutes"] is not None
                else None
            )
        }

    finally:
        connection.close()


# ============================================================
# 3. TOOL 2: SEARCH POLICY
# ============================================================

def search_policy(query):

    """
    Search the policy documents using TF-IDF retrieval.
    """

    results = retrieve(query, k=3)

    return {
        "results": results
    }


# ============================================================
# 4. TOOL 3: CALCULATE REFUND
# ============================================================

def calculate_refund(order_amount, delivery_minutes):

    """
    Calculate the refund based on the delivery time.

    More than 45 minutes: Full refund.
    45 minutes or less: No refund.
    """

    if delivery_minutes is None:
        return {"error": "delivery time unavailable"}

    if float(delivery_minutes) > 45:

        refund_amount = round(float(order_amount), 2)

    else:

        refund_amount = 0.0

    return {
        "refund_inr": refund_amount
    }


# ============================================================
# 5. HELPER: WRITE REFUND LEDGER
# ============================================================

def write_ledger(tool_name, order_id, refund_inr):

    """
    Append one JSON record to the refund ledger.
    """

    LEDGER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    record = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2)
    }

    with LEDGER_PATH.open("a", encoding="utf-8") as file:

        file.write(
            json.dumps(record) + "\n"
        )

    return {
        "refund_inr": round(float(refund_inr), 2)
    }


# ============================================================
# 6. TOOL 4: ISSUE REFUND
# ============================================================

def issue_refund(order_id, refund_inr):

    """
    Record an automatically approved refund.
    """

    return write_ledger(
        tool_name="issue_refund",
        order_id=order_id,
        refund_inr=refund_inr
    )


# ============================================================
# 7. TOOL 5: ESCALATE TO HUMAN
# ============================================================

def escalate_to_human(order_id, refund_inr):

    """
    Record a refund requiring human approval.
    """

    return write_ledger(
        tool_name="escalate_to_human",
        order_id=order_id,
        refund_inr=refund_inr
    )


# ============================================================
# 8. TOOL SPECIFICATIONS
# ============================================================

TOOLS_SPEC = [

    {
        "name": "get_order",
        "description": "Retrieve the status, amount and delivery time of an order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID to look up."
                }
            },
            "required": ["order_id"]
        }
    },

    {
        "name": "search_policy",
        "description": "Search the policy documents and return the three most relevant chunks.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The policy question or search query."
                }
            },
            "required": ["query"]
        }
    },

    {
        "name": "calculate_refund",
        "description": "Calculate the eligible refund using the order amount and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {
                    "type": "number",
                    "description": "The total order amount in INR."
                },
                "delivery_minutes": {
                    "type": "number",
                    "description": "The delivery time in minutes."
                }
            },
            "required": [
                "order_amount",
                "delivery_minutes"
            ]
        }
    },

    {
        "name": "issue_refund",
        "description": "Record an automatically approved refund in the refund ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID."
                },
                "refund_inr": {
                    "type": "number",
                    "description": "The approved refund amount in INR."
                }
            },
            "required": ["order_id", "refund_inr"]
        }
    },

    {
        "name": "escalate_to_human",
        "description": "Record a refund requiring human approval in the refund ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID."
                },
                "refund_inr": {
                    "type": "number",
                    "description": "The refund amount requiring approval in INR."
                }
            },
            "required": ["order_id", "refund_inr"]
        }
    }

]


# ============================================================
# 9. TOOL MAP
# ============================================================

TOOL_MAP = {

    "get_order": get_order,

    "search_policy": search_policy,

    "calculate_refund": calculate_refund,

    "issue_refund": issue_refund,

    "escalate_to_human": escalate_to_human

}


# ============================================================
# 10. BASIC VALIDATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PART D - REFUND TOOLS VALIDATION")
    print("=" * 60)

    print("\nTOOL SPECIFICATIONS")

    for tool in TOOLS_SPEC:
        print(
            f"{tool['name']}: {tool['description']}"
        )

    print("\nTotal tools:", len(TOOLS_SPEC))
    print("Total TOOL_MAP entries:", len(TOOL_MAP))

    assert len(TOOLS_SPEC) == 5
    assert len(TOOL_MAP) == 5

    assert set(TOOL_MAP.keys()) == {
        "get_order",
        "search_policy",
        "calculate_refund",
        "issue_refund",
        "escalate_to_human"
    }

    print("\nPASS: All five tools are registered.")

    print("\nREFUND CALCULATION TESTS")

    test_1 = calculate_refund(500, 60)
    print("60-minute delivery:", test_1)
    assert test_1["refund_inr"] == 500.0

    test_2 = calculate_refund(500, 45)
    print("45-minute delivery:", test_2)
    assert test_2["refund_inr"] == 0.0

    test_3 = calculate_refund(500, 30)
    print("30-minute delivery:", test_3)
    assert test_3["refund_inr"] == 0.0

    print("\nPASS: All refund calculation tests passed.")

    print("\nPOLICY SEARCH TEST")

    policy_result = search_policy(
        "refund policy late delivery"
    )

    assert len(policy_result["results"]) == 3

    for result in policy_result["results"]:
        print(
            result["source"],
            round(result["score"], 4)
        )

    print("\nPASS: Policy search returned three results.")

    print("\nORDER LOOKUP TEST")

    connection = sqlite3.connect(DATABASE_PATH)

    sample_order = connection.execute(
        "SELECT order_id FROM orders LIMIT 1"
    ).fetchone()

    connection.close()

    assert sample_order is not None

    order_result = get_order(sample_order[0])

    print(order_result)

    assert "order_id" in order_result
    assert "status" in order_result

    print("\nPASS: Database order lookup successful.")

    print("\nPART D STEP 4 COMPLETED.")
