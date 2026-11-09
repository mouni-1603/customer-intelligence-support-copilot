
from pathlib import Path
import sys
import time
import random

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import mock_llm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POLICY_DIR = ROOT / "data" / "policies"
EVALUATION_FILE = ROOT / "data" / "raw" / "policy_eval.csv"

CHUNK_SIZE = 40
CHUNK_OVERLAP = 10

POLICY_CHUNKS = []
VECTORIZER = None
CHUNK_MATRIX = None


# ============================================================
# 2. LOAD AND CHUNK POLICY DOCUMENTS
# ============================================================

def load_policy_chunks(overlap=CHUNK_OVERLAP):

    if overlap < 0 or overlap >= CHUNK_SIZE:
        raise ValueError("Overlap must be between 0 and 39.")

    policy_files = sorted(POLICY_DIR.glob("*.txt"))

    if len(policy_files) != 5:
        raise ValueError(
            f"Expected 5 policy files, found {len(policy_files)}."
        )

    chunks = []

    for policy_file in policy_files:

        text = policy_file.read_text(encoding="utf-8")
        words = text.split()

        start = 0

        while start < len(words):

            end = min(start + CHUNK_SIZE, len(words))

            chunks.append({
                "source": policy_file.name,
                "text": " ".join(words[start:end])
            })

            if end >= len(words):
                break

            start += CHUNK_SIZE - overlap

    return chunks


# ============================================================
# 3. BUILD TF-IDF RETRIEVAL INDEX
# ============================================================

def initialize_retriever(overlap=CHUNK_OVERLAP):

    global POLICY_CHUNKS, VECTORIZER, CHUNK_MATRIX

    POLICY_CHUNKS = load_policy_chunks(overlap)

    if not POLICY_CHUNKS:
        raise ValueError("No policy chunks were created.")

    chunk_texts = [
        chunk["text"] for chunk in POLICY_CHUNKS
    ]

    VECTORIZER = TfidfVectorizer()
    CHUNK_MATRIX = VECTORIZER.fit_transform(chunk_texts)

    return POLICY_CHUNKS


# ============================================================
# 4. RETRIEVE TOP-K POLICY CHUNKS
# ============================================================

def retrieve(query, k=3):

    if VECTORIZER is None or CHUNK_MATRIX is None:
        initialize_retriever()

    if not query.strip():
        return []

    query_vector = VECTORIZER.transform([query])

    similarities = cosine_similarity(
        query_vector, CHUNK_MATRIX
    ).flatten()

    ranked_indices = similarities.argsort()[::-1][:k]

    results = []

    for index in ranked_indices:

        chunk = POLICY_CHUNKS[index]

        results.append({
            "source": chunk["source"],
            "score": float(similarities[index]),
            "text": chunk["text"]
        })

    return results


# ============================================================
# 5. POLICY ASSISTANT
# ============================================================

POLICY_SYSTEM_PROMPT = """
You are a POLICY ASSISTANT.

Answer the user's question using only the supplied policy context.

Do not use outside knowledge or make assumptions.

If the supplied context does not contain the answer,
respond with a sentence beginning exactly:

I could not find

Be concise, accurate, and directly answer the question.
"""


def call_policy_llm(messages):

    for retry_number in range(4):

        try:
            return mock_llm.chat(messages)

        except mock_llm.TransientLLMError:

            if retry_number == 3:
                raise

            wait = (
                0.05
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"Policy LLM retry {retry_number + 1}; "
                f"waiting {wait:.5f} seconds"
            )

            time.sleep(wait)


def answer(question):

    retrieved_chunks = retrieve(question, k=3)

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"[{chunk['source']}]\n{chunk['text']}"
        )

    context = "\n".join(context_parts)

    user_message = (
        f"CONTEXT:\n"
        f"{context}\n"
        f"QUESTION:\n"
        f"{question}"
    )

    messages = [
        {
            "role": "system",
            "content": POLICY_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    response = call_policy_llm(messages)

    response_text = response.choices[0].message.content

    sources_used = list(dict.fromkeys(
        chunk["source"] for chunk in retrieved_chunks
    ))

    return {
        "answer": response_text,
        "sources": sources_used
    }


# ============================================================
# 6. EVALUATE POLICY RAG
# ============================================================

def evaluate_policy_rag(overlap):

    print("\n" + "=" * 65)
    print(f"POLICY EVALUATION - OVERLAP {overlap}")
    print("=" * 65)

    chunks = initialize_retriever(overlap=overlap)

    print("Total chunks:", len(chunks))

    mock_llm.reset()

    evaluation_df = pd.read_csv(EVALUATION_FILE)

    results = []

    for _, row in evaluation_df.iterrows():

        question = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = row["expected_fact"]

        retrieved = retrieve(question, k=3)

        retrieved_sources = [
            chunk["source"] for chunk in retrieved
        ]

        response = answer(question)
        generated_answer = response["answer"]

        if expected_doc != "NONE":

            document_hit = expected_doc in retrieved_sources

            fact_hit = (
                str(expected_fact) in generated_answer
                if pd.notna(expected_fact)
                else False
            )

            fallback_hit = None

        else:

            document_hit = None
            fact_hit = None

            fallback_hit = generated_answer.startswith(
                "I could not find"
            )

        results.append({
            "question": question,
            "expected_doc": expected_doc,
            "retrieved_sources": retrieved_sources,
            "answer": generated_answer,
            "document_hit": document_hit,
            "fact_hit": fact_hit,
            "fallback_hit": fallback_hit
        })

    results_df = pd.DataFrame(results)

    answerable = results_df[
        results_df["expected_doc"] != "NONE"
    ].copy()

    unanswerable = results_df[
        results_df["expected_doc"] == "NONE"
    ].copy()

    document_hits = int(answerable["document_hit"].sum())
    fact_hits = int(answerable["fact_hit"].sum())
    fallback_hits = int(unanswerable["fallback_hit"].sum())

    print("\nEVALUATION SUMMARY")
    print("-" * 65)

    print("Total questions:", len(results_df))
    print("Answerable questions:", len(answerable))
    print("Unanswerable questions:", len(unanswerable))

    print(
        "Expected document retrieved:",
        f"{document_hits}/{len(answerable)}"
    )

    print(
        "Expected fact found in answer:",
        f"{fact_hits}/{len(answerable)}"
    )

    print(
        "Correct fallback responses:",
        f"{fallback_hits}/{len(unanswerable)}"
    )

    print("\nQUESTION-LEVEL RESULTS")
    print("-" * 65)

    for _, result in results_df.iterrows():

        print("\nQuestion:", result["question"])
        print("Expected document:", result["expected_doc"])
        print("Retrieved sources:", result["retrieved_sources"])
        print("Answer:", result["answer"])

        if result["expected_doc"] == "NONE":

            print("Fallback correct:", result["fallback_hit"])

        else:

            print("Document retrieved:", result["document_hit"])
            print("Expected fact found:", result["fact_hit"])

    # Correct Boolean filtering:
    # eq(False) avoids the Python bitwise-NOT issue.

    failed_answerable = answerable[
        answerable["document_hit"].eq(False)
        | answerable["fact_hit"].eq(False)
    ]

    failed_unanswerable = unanswerable[
        unanswerable["fallback_hit"].eq(False)
    ]

    failed_questions = pd.concat(
        [failed_answerable, failed_unanswerable],
        ignore_index=True
    )

    print("\nFAILED QUESTIONS")
    print("-" * 65)

    if failed_questions.empty:

        print("None. All evaluation checks passed.")

    else:

        for _, failed in failed_questions.iterrows():

            print("\nQuestion:", failed["question"])
            print("Expected document:", failed["expected_doc"])
            print("Retrieved sources:", failed["retrieved_sources"])
            print("Generated answer:", failed["answer"])

    return {
        "overlap": overlap,
        "total_chunks": len(chunks),
        "document_hits": document_hits,
        "fact_hits": fact_hits,
        "fallback_hits": fallback_hits,
        "failed_questions": failed_questions,
        "results": results_df
    }


# ============================================================
# 7. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 65)
    print("PART D - POLICY CHUNKING AND RETRIEVAL")
    print("=" * 65)

    # STEP 1: CHUNK DOCUMENTS

    chunks = initialize_retriever(overlap=10)

    print("\nCHUNKING SUMMARY")
    print("-" * 65)

    print("Total policy documents:", 5)
    print("Chunk size:", CHUNK_SIZE)
    print("Chunk overlap:", CHUNK_OVERLAP)
    print("Total chunks:", len(chunks))

    counts = {}

    for chunk in chunks:

        source = chunk["source"]

        counts[source] = counts.get(source, 0) + 1

    print("\nChunks by policy document:")

    for source, count in sorted(counts.items()):
        print(f"{source}: {count}")

    expected_counts = {
        "refund_policy.txt": 4,
        "delivery_policy.txt": 3,
        "escalation_policy.txt": 3,
        "payment_policy.txt": 3,
        "privacy_policy.txt": 2
    }

    assert len(chunks) == 15
    assert counts == expected_counts

    print("\nPASS: Total chunk count is 15.")
    print("PASS: All five document chunk counts match.")

    # STEP 2: TEST RETRIEVAL

    query = "refund policy late delivery"

    print("\nRETRIEVAL TEST")
    print("-" * 65)
    print("Query:", query)

    retrieval_results = retrieve(query, k=3)

    for rank, result in enumerate(retrieval_results, start=1):

        print(f"\nRank {rank}")
        print("Source:", result["source"])
        print("Similarity score:", round(result["score"], 4))
        print("Text:", result["text"])

    assert len(retrieval_results) == 3

    assert all(
        result["source"] == "refund_policy.txt"
        for result in retrieval_results
    )

    print("\nPASS: Top three results are from refund_policy.txt.")

    # STEP 3: TEST POLICY ASSISTANT

    print("\nPOLICY ASSISTANT TEST")
    print("-" * 65)

    test_questions = [
        "What is the refund rule for late delivery?",
        "What happens if a customer requests a refund for a cancelled order?",
        "What is the refund approval limit?"
    ]

    mock_llm.reset()

    for question in test_questions:

        result = answer(question)

        print("\nQuestion:", question)
        print("Answer:", result["answer"])
        print("Sources:", result["sources"])

    # STEP 4: EVALUATE WITH 10-WORD OVERLAP

    overlap_results = evaluate_policy_rag(overlap=10)

    # STEP 5: EVALUATE WITH ZERO OVERLAP

    zero_overlap_results = evaluate_policy_rag(overlap=0)

    # STEP 6: COMPARE RESULTS

    print("\n" + "=" * 65)
    print("OVERLAP VS ZERO-OVERLAP COMPARISON")
    print("=" * 65)

    comparison = pd.DataFrame([
        {
            "Chunking": "40 words, 10 overlap",
            "Total chunks": overlap_results["total_chunks"],
            "Document hits": overlap_results["document_hits"],
            "Fact hits": overlap_results["fact_hits"],
            "Fallback hits": overlap_results["fallback_hits"]
        },
        {
            "Chunking": "40 words, 0 overlap",
            "Total chunks": zero_overlap_results["total_chunks"],
            "Document hits": zero_overlap_results["document_hits"],
            "Fact hits": zero_overlap_results["fact_hits"],
            "Fallback hits": zero_overlap_results["fallback_hits"]
        }
    ])

    print(comparison.to_string(index=False))

    # STEP 7: ZERO-OVERLAP FAILURE ANALYSIS

    print("\nZERO-OVERLAP FAILURE ANALYSIS")
    print("-" * 65)

    failed_zero = zero_overlap_results["failed_questions"]

    if failed_zero.empty:

        print(
            "No failures detected. Compare these results "
            "with the assignment reference and inspect "
            "the policy chunks if necessary."
        )

    else:

        for _, failed in failed_zero.iterrows():

            print("\nQuestion:", failed["question"])
            print("Expected document:", failed["expected_doc"])
            print("Retrieved sources:", failed["retrieved_sources"])
            print("Answer:", failed["answer"])

        print(
            "\nExplanation: Without overlap, a policy rule "
            "can be split between consecutive chunks. "
            "The retrieval system may not find the complete "
            "information in its top three chunks."
        )

    # STEP 8: RESTORE NORMAL RETRIEVAL INDEX

    initialize_retriever(overlap=10)

    print("\nNORMAL RETRIEVAL INDEX RESTORED.")

    # STEP 9: FINAL SUMMARY

    print("\n" + "=" * 65)
    print("PART D STEP 3 FINAL SUMMARY")
    print("=" * 65)

    print(
        "10-word overlap document hits:",
        f"{overlap_results['document_hits']}/12"
    )

    print(
        "10-word overlap fact hits:",
        f"{overlap_results['fact_hits']}/12"
    )

    print(
        "10-word overlap fallback hits:",
        f"{overlap_results['fallback_hits']}/2"
    )

    print(
        "Zero-overlap failed questions:",
        len(zero_overlap_results["failed_questions"])
    )

    print("\nPART D STEP 3 COMPLETED")


# ============================================================
# 8. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
