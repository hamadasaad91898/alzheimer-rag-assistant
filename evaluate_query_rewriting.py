import os
import json
import math
import re

from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client


load_dotenv(override=True)


# =========================================================
# Config
# =========================================================

azure_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

embedding_model = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
)

chat_model = os.getenv(
    "AZURE_CHAT_DEPLOYMENT"
)

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")


# =========================================================
# Clients
# =========================================================

openai_client = OpenAI(
    api_key=azure_key,
    base_url=azure_endpoint.rstrip("/") + "/openai/v1/"
)

supabase = create_client(
    supabase_url,
    supabase_key
)


# =========================================================
# Settings
# =========================================================

VECTOR_K = 5
CANDIDATE_K = 10
FINAL_K = 5

RERANK_MAX_ATTEMPTS = 2

# GPT-5.6 Sol is a reasoning model.
# Give it enough output budget to finish the ranking.
RERANK_MAX_OUTPUT_TOKENS = 1200

QUERY_REWRITE_MAX_OUTPUT_TOKENS = 500


# =========================================================
# JSON Helper
# =========================================================

def parse_json_response(text):
    text = text.strip()

    try:
        data = json.loads(text)

        if isinstance(data, dict):
            return data

    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "Model did not return valid JSON."
        )

    data = json.loads(
        text[start:end + 1]
    )

    if not isinstance(data, dict):
        raise ValueError(
            "JSON response is not an object."
        )

    return data


# =========================================================
# Embeddings
# =========================================================

def create_embedding(text):
    response = openai_client.embeddings.create(
        model=embedding_model,
        input=text
    )

    return response.data[0].embedding


# =========================================================
# Vector Search
# =========================================================

def vector_search(query, top_k):
    embedding = create_embedding(
        query
    )

    response = supabase.rpc(
        "match_documents",
        {
            "query_embedding": embedding,
            "match_count": top_k
        }
    ).execute()

    return response.data or []


# =========================================================
# Query Rewriting
# =========================================================

QUERY_REWRITE_PROMPT = """
You are a query rewriter for an Alzheimer's disease
Retrieval-Augmented Generation system.

Your ONLY task is to rewrite the user's question
into a clear English retrieval query.

Do not answer the question.

Rules:

1. Preserve the original intent exactly.

2. Do not add new topics.

3. Do not add unsupported facts.

4. If the question is Arabic, translate it into
   clear English.

5. If the question is English and already clear,
   preserve it or make only minimal changes.

6. Do not turn diagnosis into treatment.

7. Do not turn risk factors into treatment.

8. Do not turn lifestyle questions into medication
   management questions.

9. Do not turn a general question into a
   patient-specific question.

10. Keep important medical terms unchanged whenever
    possible.

11. Keep the rewritten query concise and standalone.

Return valid JSON only.

Format:

{
  "rewritten_query": "..."
}
""".strip()


def rewrite_query(question):
    response = openai_client.responses.create(
        model=chat_model,
        instructions=QUERY_REWRITE_PROMPT,
        input=f"""
Original user question:

{question}
""".strip(),
        max_output_tokens=QUERY_REWRITE_MAX_OUTPUT_TOKENS
    )

    raw_output = (
        response.output_text or ""
    ).strip()

    if not raw_output:
        raise ValueError(
            "Query rewriter returned empty output."
        )

    data = parse_json_response(
        raw_output
    )

    rewritten_query = data.get(
        "rewritten_query"
    )

    if not isinstance(
        rewritten_query,
        str
    ):
        raise ValueError(
            "Missing rewritten_query."
        )

    rewritten_query = rewritten_query.strip()

    if not rewritten_query:
        raise ValueError(
            "Empty rewritten query."
        )

    return rewritten_query


# =========================================================
# GPT Reranker
# =========================================================

RERANK_PROMPT = """
You are a strict passage reranker for an
Alzheimer's disease Retrieval-Augmented
Generation system.

Your ONLY task is to rank the candidate passages.

Do not answer the medical question.

Rank according to the ORIGINAL user question.

Ranking priority:

1. Direct relevance to the exact question.

2. Specific passages above broad passages.

3. Passages containing enough information to answer
   the question above passages that only mention
   related terms.

4. Preserve the original user intent.

5. If two passages are equally relevant, prefer the
   passage with higher Vector Similarity.

Rules:

- Use only the provided passages.
- Do not use outside knowledge.
- Do not explain your ranking.
- Do not invent Chunk IDs.
- Include every candidate Chunk ID exactly once.

OUTPUT RULE:

Return ONLY the Chunk IDs separated by commas.

Example:

12,13,8,10,9

Do not return JSON.
Do not return brackets.
Do not return markdown.
Do not return explanations.
Do not return any other text.
""".strip()


def build_rerank_input(
    original_question,
    rewritten_query,
    candidates
):
    passages = []

    for index, chunk in enumerate(
        candidates,
        start=1
    ):
        similarity = chunk.get(
            "similarity",
            0
        )

        passages.append(
            f"""
PASSAGE {index}

Chunk ID: {chunk["chunk_id"]}
Section: {chunk["section"]}
Pages: {chunk.get("pages", [])}
Vector Similarity: {similarity:.6f}

Content:
{chunk["content"]}
""".strip()
        )

    joined_passages = "\n\n---\n\n".join(
        passages
    )

    return f"""
ORIGINAL USER QUESTION:

{original_question}


REWRITTEN RETRIEVAL QUERY:

{rewritten_query}


CANDIDATE PASSAGES:

{joined_passages}
""".strip()


def parse_reranker_ids(
    text,
    valid_ids
):
    text = (
        text or ""
    ).strip()

    if not text:
        raise ValueError(
            "Reranker returned empty output."
        )

    numbers = re.findall(
        r"\d+",
        text
    )

    if not numbers:
        raise ValueError(
            "No Chunk IDs found in reranker output."
        )

    parsed_ids = []

    for value in numbers:
        chunk_id = int(value)

        if (
            chunk_id in valid_ids
            and chunk_id not in parsed_ids
        ):
            parsed_ids.append(
                chunk_id
            )

    missing_ids = [
        chunk_id
        for chunk_id in valid_ids
        if chunk_id not in parsed_ids
    ]

    if missing_ids:
        raise ValueError(
            f"Reranker missed IDs: {missing_ids}"
        )

    if len(parsed_ids) != len(valid_ids):
        raise ValueError(
            "Invalid number of ranked IDs."
        )

    return parsed_ids


def rerank_chunks(
    original_question,
    rewritten_query,
    candidates
):
    if not candidates:
        return []

    valid_ids = [
        int(chunk["chunk_id"])
        for chunk in candidates
    ]

    rerank_input = build_rerank_input(
        original_question,
        rewritten_query,
        candidates
    )

    last_error = None

    for attempt in range(
        1,
        RERANK_MAX_ATTEMPTS + 1
    ):
        raw_output = ""

        try:
            instructions = RERANK_PROMPT

            if attempt > 1:
                instructions += f"""

IMPORTANT RETRY:

Your previous response could not be used.

Return exactly these Chunk IDs,
each exactly once:

{",".join(str(x) for x in valid_ids)}

Rank them by relevance.

Return ONLY comma-separated Chunk IDs.

No explanation.
No JSON.
No brackets.
""".strip()

            response = openai_client.responses.create(
                model=chat_model,
                instructions=instructions,
                input=rerank_input,
                max_output_tokens=RERANK_MAX_OUTPUT_TOKENS
            )

            raw_output = (
                response.output_text or ""
            ).strip()

            ranked_ids = parse_reranker_ids(
                raw_output,
                valid_ids
            )

            chunk_map = {
                int(chunk["chunk_id"]): chunk
                for chunk in candidates
            }

            reranked = [
                chunk_map[chunk_id]
                for chunk_id in ranked_ids
            ]

            return reranked[:FINAL_K]

        except Exception as error:
            last_error = error

            print()
            print(
                f"Reranker attempt "
                f"{attempt} failed:"
            )

            print(
                error
            )

            print(
                "Raw reranker output:"
            )

            if raw_output:
                print(
                    repr(raw_output)
                )
            else:
                print(
                    "<EMPTY>"
                )

    raise ValueError(
        "Reranker failed after "
        f"{RERANK_MAX_ATTEMPTS} attempts: "
        f"{last_error}"
    )


# =========================================================
# Metrics
# =========================================================

def evaluate_question(
    retrieved_ids,
    relevant_ids,
    k=5
):
    retrieved_ids = retrieved_ids[:k]

    relevant = set(
        relevant_ids
    )

    hits = [
        chunk_id
        for chunk_id in retrieved_ids
        if chunk_id in relevant
    ]

    hit_rate = (
        1.0
        if hits
        else 0.0
    )

    precision = (
        len(hits) / k
    )

    recall = (
        len(hits) / len(relevant)
        if relevant
        else 0.0
    )

    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1
    ):
        if chunk_id in relevant:
            reciprocal_rank = (
                1.0 / rank
            )
            break

    dcg = 0.0

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1
    ):
        relevance = (
            1.0
            if chunk_id in relevant
            else 0.0
        )

        dcg += (
            relevance
            / math.log2(rank + 1)
        )

    ideal_hits = min(
        len(relevant),
        k
    )

    idcg = sum(
        1.0 / math.log2(
            rank + 1
        )
        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    ndcg = (
        dcg / idcg
        if idcg > 0
        else 0.0
    )

    return {
        "hit_rate": hit_rate,
        "precision": precision,
        "recall": recall,
        "mrr": reciprocal_rank,
        "ndcg": ndcg
    }


def average_metrics(results):
    count = len(results)

    return {
        "hit_rate_at_5": sum(
            item["hit_rate"]
            for item in results
        ) / count,

        "precision_at_5": sum(
            item["precision"]
            for item in results
        ) / count,

        "recall_at_5": sum(
            item["recall"]
            for item in results
        ) / count,

        "mrr": sum(
            item["mrr"]
            for item in results
        ) / count,

        "ndcg_at_5": sum(
            item["ndcg"]
            for item in results
        ) / count
    }


# =========================================================
# Print Metrics
# =========================================================

def print_metrics(
    title,
    metrics
):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        f"Hit Rate@5:        "
        f"{metrics['hit_rate_at_5']:.4f}"
    )

    print(
        f"Mean Precision@5:  "
        f"{metrics['precision_at_5']:.4f}"
    )

    print(
        f"Mean Recall@5:     "
        f"{metrics['recall_at_5']:.4f}"
    )

    print(
        f"MRR:               "
        f"{metrics['mrr']:.4f}"
    )

    print(
        f"Mean nDCG@5:       "
        f"{metrics['ndcg_at_5']:.4f}"
    )


# =========================================================
# Main Evaluation
# =========================================================

def main():

    with open(
        "eval_questions.json",
        "r",
        encoding="utf-8"
    ) as file:
        questions = json.load(
            file
        )

    original_results = []
    rewrite_results = []
    rewrite_rerank_results = []

    detailed_results = []

    reranker_failures = 0

    total = len(
        questions
    )

    for index, item in enumerate(
        questions,
        start=1
    ):

        question = item[
            "question"
        ]

        relevant_ids = item[
            "relevant_chunk_ids"
        ]

        print()
        print("=" * 70)
        print(
            f"QUESTION {index}/{total}"
        )
        print("=" * 70)

        print(
            question
        )

        # =================================================
        # 1. Original Vector Search
        # =================================================

        original_chunks = vector_search(
            question,
            VECTOR_K
        )

        original_ids = [
            int(chunk["chunk_id"])
            for chunk in original_chunks
        ]

        original_metrics = evaluate_question(
            original_ids,
            relevant_ids
        )

        original_results.append(
            original_metrics
        )

        # =================================================
        # 2. Query Rewrite
        # =================================================

        try:
            rewritten_query = rewrite_query(
                question
            )

        except Exception as error:
            print()
            print(
                "Rewrite error:",
                error
            )

            print(
                "Using original question."
            )

            rewritten_query = question

        print()
        print(
            "Original Query:"
        )

        print(
            question
        )

        print()
        print(
            "Rewritten Query:"
        )

        print(
            rewritten_query
        )

        # =================================================
        # 3. Rewrite + Vector
        # =================================================

        rewritten_chunks = vector_search(
            rewritten_query,
            VECTOR_K
        )

        rewritten_ids = [
            int(chunk["chunk_id"])
            for chunk in rewritten_chunks
        ]

        rewrite_metrics = evaluate_question(
            rewritten_ids,
            relevant_ids
        )

        rewrite_results.append(
            rewrite_metrics
        )

        # =================================================
        # 4. Rewrite + Top10 + Reranker
        # =================================================

        candidates = vector_search(
            rewritten_query,
            CANDIDATE_K
        )

        fallback_used = False

        try:
            reranked_chunks = rerank_chunks(
                question,
                rewritten_query,
                candidates
            )

        except Exception as error:
            reranker_failures += 1
            fallback_used = True

            print()
            print(
                "Reranker final error:"
            )

            print(
                error
            )

            print(
                "Using Vector Top 5 fallback."
            )

            reranked_chunks = candidates[
                :FINAL_K
            ]

        reranked_ids = [
            int(chunk["chunk_id"])
            for chunk in reranked_chunks
        ]

        rerank_metrics = evaluate_question(
            reranked_ids,
            relevant_ids
        )

        rewrite_rerank_results.append(
            rerank_metrics
        )

        print()
        print(
            f"Relevant:         "
            f"{relevant_ids}"
        )

        print(
            f"Original Vector:  "
            f"{original_ids}"
        )

        print(
            f"Rewrite Vector:   "
            f"{rewritten_ids}"
        )

        print(
            f"Rewrite+Reranker: "
            f"{reranked_ids}"
        )

        print(
            f"Fallback Used:    "
            f"{fallback_used}"
        )

        detailed_results.append(
            {
                "question":
                    question,

                "rewritten_query":
                    rewritten_query,

                "relevant_chunk_ids":
                    relevant_ids,

                "original_vector_ids":
                    original_ids,

                "rewrite_vector_ids":
                    rewritten_ids,

                "rewrite_reranker_ids":
                    reranked_ids,

                "reranker_fallback_used":
                    fallback_used,

                "original_metrics":
                    original_metrics,

                "rewrite_metrics":
                    rewrite_metrics,

                "rewrite_reranker_metrics":
                    rerank_metrics
            }
        )

    # =====================================================
    # Final Metrics
    # =====================================================

    original_summary = average_metrics(
        original_results
    )

    rewrite_summary = average_metrics(
        rewrite_results
    )

    rewrite_rerank_summary = average_metrics(
        rewrite_rerank_results
    )

    print_metrics(
        "1. ORIGINAL VECTOR SEARCH",
        original_summary
    )

    print_metrics(
        "2. QUERY REWRITE + VECTOR",
        rewrite_summary
    )

    print_metrics(
        "3. QUERY REWRITE + VECTOR + GPT RERANKER",
        rewrite_rerank_summary
    )

    # =====================================================
    # Reliability
    # =====================================================

    success_count = (
        total - reranker_failures
    )

    success_rate = (
        success_count / total
        if total
        else 0.0
    )

    print()
    print("=" * 70)
    print("RERANKER RELIABILITY")
    print("=" * 70)

    print(
        f"Total Questions:       "
        f"{total}"
    )

    print(
        f"Final Failures:        "
        f"{reranker_failures}"
    )

    print(
        f"Vector Fallbacks Used: "
        f"{reranker_failures}"
    )

    print(
        f"Reranker Success Rate: "
        f"{success_rate:.4f}"
    )

    # =====================================================
    # Save Results
    # =====================================================

    output = {
        "settings": {
            "vector_k":
                VECTOR_K,

            "candidate_k":
                CANDIDATE_K,

            "final_k":
                FINAL_K,

            "rerank_max_attempts":
                RERANK_MAX_ATTEMPTS,

            "rerank_max_output_tokens":
                RERANK_MAX_OUTPUT_TOKENS,

            "embedding_model":
                embedding_model,

            "reranker_model":
                chat_model,

            "reranker_output_format":
                "comma_separated_chunk_ids"
        },

        "original_vector":
            original_summary,

        "query_rewrite_vector":
            rewrite_summary,

        "query_rewrite_vector_reranker":
            rewrite_rerank_summary,

        "reranker_reliability": {
            "total_questions":
                total,

            "final_failures":
                reranker_failures,

            "fallbacks":
                reranker_failures,

            "success_rate":
                success_rate
        },

        "questions":
            detailed_results
    }

    with open(
        "query_rewriting_evaluation.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("=" * 70)

    print(
        "Saved to: "
        "query_rewriting_evaluation.json"
    )

    print("=" * 70)


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    main()