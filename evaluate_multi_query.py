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

FINAL_K = 5

# Retrieve enough candidates from each query
TOP_PER_QUERY = 10

# Multi-query:
# 1 normal rewritten query
# + 2 alternative queries
MULTI_QUERY_COUNT = 3

QUERY_REWRITE_MAX_OUTPUT_TOKENS = 500
MULTI_QUERY_MAX_OUTPUT_TOKENS = 800

MULTI_QUERY_MAX_ATTEMPTS = 2


# =========================================================
# JSON Helper
# =========================================================

def parse_json_response(text):
    text = (text or "").strip()

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
# Normal Query Rewriting
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

3. Do not add facts or assumptions.

4. If the question is Arabic, translate it
   into clear English.

5. If the question is English and already clear,
   preserve it or make only minimal changes.

6. Do not turn diagnosis into treatment.

7. Do not turn risk factors into treatment.

8. Do not turn lifestyle questions into
   medication-management questions.

9. Do not turn a general question into a
   patient-specific question.

10. Preserve important medical entities
    whenever possible.

11. Keep the query concise and standalone.

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

    rewritten_query = (
        rewritten_query.strip()
    )

    if not rewritten_query:
        raise ValueError(
            "Empty rewritten query."
        )

    return rewritten_query


# =========================================================
# Multi-Query Generation
# =========================================================

MULTI_QUERY_PROMPT = """
You generate alternative retrieval queries for an
Alzheimer's disease RAG system.

You are NOT answering the user's question.

You will receive:

1. The original user question.
2. A primary rewritten English retrieval query.

Generate exactly TWO alternative English retrieval queries.

The alternatives must express the SAME information need.

The purpose is to improve semantic retrieval against
a scientific Alzheimer's disease document.

Rules:

1. Preserve the original intent exactly.

2. Do not make the query broader.

3. Do not make the query narrower in a way that
   removes part of the user's intent.

4. Do not add symptoms, diagnoses, treatments,
   medications, risk factors, mechanisms,
   recommendations, or facts that were not requested.

5. Use alternative scientific wording when useful.

6. Preserve important entities such as:
   tau
   amyloid-beta
   APOE epsilon 4
   p-tau217
   memantine
   donepezil
   lecanemab
   donanemab
   MRI
   amyloid PET
   tau PET

7. Do not answer the question.

8. Both alternatives must be concise and standalone.

OUTPUT FORMAT:

ALT_1: first alternative query
ALT_2: second alternative query

Return only these two lines.
""".strip()


def parse_multi_query_output(text):
    text = (
        text or ""
    ).strip()

    if not text:
        raise ValueError(
            "Multi-query generator returned empty output."
        )

    alt_1_match = re.search(
        r"ALT_1\s*:\s*(.+)",
        text,
        flags=re.IGNORECASE
    )

    alt_2_match = re.search(
        r"ALT_2\s*:\s*(.+)",
        text,
        flags=re.IGNORECASE
    )

    if (
        not alt_1_match
        or not alt_2_match
    ):
        raise ValueError(
            "Could not parse ALT_1 and ALT_2."
        )

    alt_1 = (
        alt_1_match
        .group(1)
        .strip()
    )

    alt_2 = (
        alt_2_match
        .group(1)
        .strip()
    )

    if not alt_1 or not alt_2:
        raise ValueError(
            "One or more alternative queries are empty."
        )

    return alt_1, alt_2


def generate_multi_queries(
    original_question,
    rewritten_query
):
    last_error = None

    for attempt in range(
        1,
        MULTI_QUERY_MAX_ATTEMPTS + 1
    ):
        try:
            instructions = MULTI_QUERY_PROMPT

            if attempt > 1:
                instructions += """

IMPORTANT RETRY:

Return exactly:

ALT_1: ...
ALT_2: ...

No explanation.
No markdown.
No extra text.
""".strip()

            response = openai_client.responses.create(
                model=chat_model,
                instructions=instructions,
                input=f"""
Original user question:

{original_question}


Primary rewritten retrieval query:

{rewritten_query}
""".strip(),
                max_output_tokens=MULTI_QUERY_MAX_OUTPUT_TOKENS
            )

            raw_output = (
                response.output_text or ""
            ).strip()

            alt_1, alt_2 = (
                parse_multi_query_output(
                    raw_output
                )
            )

            queries = [
                rewritten_query,
                alt_1,
                alt_2
            ]

            # Remove exact duplicates while
            # preserving order.
            unique_queries = []

            for query in queries:
                normalized = (
                    query.strip().lower()
                )

                if not any(
                    existing.strip().lower()
                    == normalized
                    for existing
                    in unique_queries
                ):
                    unique_queries.append(
                        query.strip()
                    )

            return unique_queries

        except Exception as error:
            last_error = error

            print(
                f"Multi-query attempt "
                f"{attempt} failed: "
                f"{error}"
            )

    raise ValueError(
        "Multi-query generation failed after "
        f"{MULTI_QUERY_MAX_ATTEMPTS} attempts: "
        f"{last_error}"
    )


# =========================================================
# Multi-Query Retrieval
# =========================================================

def multi_query_search(queries):
    """
    Search each query independently.

    If the same chunk appears multiple times,
    keep its highest REAL cosine similarity.

    No artificial score adjustment is used.
    """

    merged = {}

    for query_index, query in enumerate(
        queries,
        start=1
    ):
        results = vector_search(
            query,
            TOP_PER_QUERY
        )

        for chunk in results:
            chunk_id = int(
                chunk["chunk_id"]
            )

            similarity = float(
                chunk.get(
                    "similarity",
                    0
                )
            )

            if chunk_id not in merged:
                item = dict(
                    chunk
                )

                item[
                    "chunk_id"
                ] = chunk_id

                item[
                    "similarity"
                ] = similarity

                item[
                    "best_query"
                ] = query

                item[
                    "best_query_index"
                ] = query_index

                item[
                    "query_scores"
                ] = {
                    str(query_index):
                        similarity
                }

                merged[
                    chunk_id
                ] = item

            else:
                merged[
                    chunk_id
                ][
                    "query_scores"
                ][
                    str(query_index)
                ] = similarity

                current_best = float(
                    merged[
                        chunk_id
                    ].get(
                        "similarity",
                        0
                    )
                )

                if similarity > current_best:
                    merged[
                        chunk_id
                    ][
                        "similarity"
                    ] = similarity

                    merged[
                        chunk_id
                    ][
                        "best_query"
                    ] = query

                    merged[
                        chunk_id
                    ][
                        "best_query_index"
                    ] = query_index

    ranked = sorted(
        merged.values(),
        key=lambda item: float(
            item.get(
                "similarity",
                0
            )
        ),
        reverse=True
    )

    return ranked


# =========================================================
# Metrics
# =========================================================

def evaluate_question(
    retrieved_ids,
    relevant_ids,
    k=5
):
    retrieved_ids = (
        retrieved_ids[:k]
    )

    relevant = set(
        int(x)
        for x in relevant_ids
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
        1.0
        / math.log2(rank + 1)
        for rank
        in range(
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
    count = len(
        results
    )

    return {
        "hit_rate_at_5":
            sum(
                x["hit_rate"]
                for x in results
            ) / count,

        "precision_at_5":
            sum(
                x["precision"]
                for x in results
            ) / count,

        "recall_at_5":
            sum(
                x["recall"]
                for x in results
            ) / count,

        "mrr":
            sum(
                x["mrr"]
                for x in results
            ) / count,

        "ndcg_at_5":
            sum(
                x["ndcg"]
                for x in results
            ) / count
    }


# =========================================================
# Similarity Helpers
# =========================================================

def get_top_similarity(chunks):
    if not chunks:
        return 0.0

    return max(
        float(
            chunk.get(
                "similarity",
                0
            )
        )
        for chunk in chunks
    )


def get_best_relevant_similarity(
    chunks,
    relevant_ids
):
    relevant_ids = set(
        int(x)
        for x in relevant_ids
    )

    scores = [
        float(
            chunk.get(
                "similarity",
                0
            )
        )
        for chunk in chunks
        if int(
            chunk["chunk_id"]
        ) in relevant_ids
    ]

    if not scores:
        return None

    return max(
        scores
    )


# =========================================================
# Printing Helpers
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
# Main
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
    multi_results = []

    detailed_results = []

    original_top_scores = []
    rewrite_top_scores = []
    multi_top_scores = []

    original_relevant_scores = []
    rewrite_relevant_scores = []
    multi_relevant_scores = []

    relevant_score_improved = 0
    top_score_improved = 0

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
        # 1. Original Query
        # =================================================

        original_chunks = vector_search(
            question,
            TOP_PER_QUERY
        )

        original_top5 = (
            original_chunks[:FINAL_K]
        )

        original_ids = [
            int(chunk["chunk_id"])
            for chunk in original_top5
        ]

        original_metrics = (
            evaluate_question(
                original_ids,
                relevant_ids
            )
        )

        original_results.append(
            original_metrics
        )

        original_top_score = (
            get_top_similarity(
                original_chunks
            )
        )

        original_relevant_score = (
            get_best_relevant_similarity(
                original_chunks,
                relevant_ids
            )
        )

        original_top_scores.append(
            original_top_score
        )

        if original_relevant_score is not None:
            original_relevant_scores.append(
                original_relevant_score
            )

        # =================================================
        # 2. Normal Rewritten Query
        # =================================================

        try:
            rewritten_query = rewrite_query(
                question
            )

        except Exception as error:
            print(
                "Rewrite error:",
                error
            )

            rewritten_query = (
                question
            )

        rewritten_chunks = vector_search(
            rewritten_query,
            TOP_PER_QUERY
        )

        rewritten_top5 = (
            rewritten_chunks[:FINAL_K]
        )

        rewritten_ids = [
            int(chunk["chunk_id"])
            for chunk in rewritten_top5
        ]

        rewrite_metrics = (
            evaluate_question(
                rewritten_ids,
                relevant_ids
            )
        )

        rewrite_results.append(
            rewrite_metrics
        )

        rewrite_top_score = (
            get_top_similarity(
                rewritten_chunks
            )
        )

        rewrite_relevant_score = (
            get_best_relevant_similarity(
                rewritten_chunks,
                relevant_ids
            )
        )

        rewrite_top_scores.append(
            rewrite_top_score
        )

        if rewrite_relevant_score is not None:
            rewrite_relevant_scores.append(
                rewrite_relevant_score
            )

        # =================================================
        # 3. Multi-Query Generation
        # =================================================

        try:
            multi_queries = (
                generate_multi_queries(
                    question,
                    rewritten_query
                )
            )

        except Exception as error:
            print(
                "Multi-query generation error:",
                error
            )

            multi_queries = [
                rewritten_query
            ]

        # =================================================
        # 4. Multi-Query Retrieval
        # =================================================

        multi_chunks = multi_query_search(
            multi_queries
        )

        multi_top5 = (
            multi_chunks[:FINAL_K]
        )

        multi_ids = [
            int(chunk["chunk_id"])
            for chunk in multi_top5
        ]

        multi_metrics = (
            evaluate_question(
                multi_ids,
                relevant_ids
            )
        )

        multi_results.append(
            multi_metrics
        )

        multi_top_score = (
            get_top_similarity(
                multi_chunks
            )
        )

        multi_relevant_score = (
            get_best_relevant_similarity(
                multi_chunks,
                relevant_ids
            )
        )

        multi_top_scores.append(
            multi_top_score
        )

        if multi_relevant_score is not None:
            multi_relevant_scores.append(
                multi_relevant_score
            )

        # =================================================
        # Score Improvement
        # =================================================

        if (
            multi_top_score
            > rewrite_top_score + 0.000001
        ):
            top_score_improved += 1

        if (
            rewrite_relevant_score is not None
            and multi_relevant_score is not None
            and multi_relevant_score
            > rewrite_relevant_score + 0.000001
        ):
            relevant_score_improved += 1

        # =================================================
        # Print
        # =================================================

        print()
        print(
            f"Relevant IDs: "
            f"{relevant_ids}"
        )

        print()
        print(
            f"Original Query:"
        )

        print(
            question
        )

        print(
            f"Top Similarity: "
            f"{original_top_score:.4f}"
        )

        print(
            f"Best Relevant Similarity: "
            f"{original_relevant_score:.4f}"
            if original_relevant_score is not None
            else
            "Best Relevant Similarity: NOT FOUND"
        )

        print()
        print(
            "Single Rewritten Query:"
        )

        print(
            rewritten_query
        )

        print(
            f"Top Similarity: "
            f"{rewrite_top_score:.4f}"
        )

        print(
            f"Best Relevant Similarity: "
            f"{rewrite_relevant_score:.4f}"
            if rewrite_relevant_score is not None
            else
            "Best Relevant Similarity: NOT FOUND"
        )

        print()
        print(
            "Multi Queries:"
        )

        for query_index, query in enumerate(
            multi_queries,
            start=1
        ):
            print(
                f"Q{query_index}: {query}"
            )

        print()
        print(
            f"Multi Top Similarity: "
            f"{multi_top_score:.4f}"
        )

        print(
            f"Multi Best Relevant Similarity: "
            f"{multi_relevant_score:.4f}"
            if multi_relevant_score is not None
            else
            "Multi Best Relevant Similarity: NOT FOUND"
        )

        print()
        print(
            f"Original Top5: "
            f"{original_ids}"
        )

        print(
            f"Rewrite Top5:  "
            f"{rewritten_ids}"
        )

        print(
            f"Multi Top5:    "
            f"{multi_ids}"
        )

        if multi_top5:
            best_chunk = (
                multi_top5[0]
            )

            print()
            print(
                "Best Multi-Query Match:"
            )

            print(
                f"Chunk: "
                f"{best_chunk['chunk_id']}"
            )

            print(
                f"Score: "
                f"{float(best_chunk['similarity']):.4f}"
            )

            print(
                f"Best Query: "
                f"{best_chunk.get('best_query')}"
            )

        # =================================================
        # Save Detail
        # =================================================

        detailed_results.append(
            {
                "question":
                    question,

                "relevant_chunk_ids":
                    relevant_ids,

                "rewritten_query":
                    rewritten_query,

                "multi_queries":
                    multi_queries,

                "original": {
                    "top5_ids":
                        original_ids,

                    "top_similarity":
                        original_top_score,

                    "best_relevant_similarity":
                        original_relevant_score,

                    "metrics":
                        original_metrics
                },

                "single_rewrite": {
                    "top5_ids":
                        rewritten_ids,

                    "top_similarity":
                        rewrite_top_score,

                    "best_relevant_similarity":
                        rewrite_relevant_score,

                    "metrics":
                        rewrite_metrics
                },

                "multi_query": {
                    "top5_ids":
                        multi_ids,

                    "top_similarity":
                        multi_top_score,

                    "best_relevant_similarity":
                        multi_relevant_score,

                    "metrics":
                        multi_metrics
                }
            }
        )

    # =====================================================
    # Final Metrics
    # =====================================================

    original_summary = (
        average_metrics(
            original_results
        )
    )

    rewrite_summary = (
        average_metrics(
            rewrite_results
        )
    )

    multi_summary = (
        average_metrics(
            multi_results
        )
    )

    print_metrics(
        "1. ORIGINAL VECTOR SEARCH",
        original_summary
    )

    print_metrics(
        "2. SINGLE QUERY REWRITE + VECTOR",
        rewrite_summary
    )

    print_metrics(
        "3. MULTI-QUERY + MAX REAL SIMILARITY",
        multi_summary
    )

    # =====================================================
    # Similarity Summary
    # =====================================================

    avg_original_top = (
        sum(original_top_scores)
        / len(original_top_scores)
    )

    avg_rewrite_top = (
        sum(rewrite_top_scores)
        / len(rewrite_top_scores)
    )

    avg_multi_top = (
        sum(multi_top_scores)
        / len(multi_top_scores)
    )

    avg_original_relevant = (
        sum(original_relevant_scores)
        / len(original_relevant_scores)
        if original_relevant_scores
        else 0.0
    )

    avg_rewrite_relevant = (
        sum(rewrite_relevant_scores)
        / len(rewrite_relevant_scores)
        if rewrite_relevant_scores
        else 0.0
    )

    avg_multi_relevant = (
        sum(multi_relevant_scores)
        / len(multi_relevant_scores)
        if multi_relevant_scores
        else 0.0
    )

    print()
    print("=" * 70)
    print("SIMILARITY SCORE COMPARISON")
    print("=" * 70)

    print(
        f"Average Original Top Score:       "
        f"{avg_original_top:.4f}"
    )

    print(
        f"Average Single Rewrite Top Score: "
        f"{avg_rewrite_top:.4f}"
    )

    print(
        f"Average Multi-Query Top Score:    "
        f"{avg_multi_top:.4f}"
    )

    print()

    print(
        f"Average Original Relevant Score:       "
        f"{avg_original_relevant:.4f}"
    )

    print(
        f"Average Single Rewrite Relevant Score: "
        f"{avg_rewrite_relevant:.4f}"
    )

    print(
        f"Average Multi-Query Relevant Score:    "
        f"{avg_multi_relevant:.4f}"
    )

    print()

    print(
        f"Top Score Improved: "
        f"{top_score_improved}/{total}"
    )

    print(
        f"Relevant Score Improved: "
        f"{relevant_score_improved}/{total}"
    )

    top_score_gain = (
        avg_multi_top
        - avg_rewrite_top
    )

    relevant_score_gain = (
        avg_multi_relevant
        - avg_rewrite_relevant
    )

    print()

    print(
        f"Average Top Score Gain: "
        f"{top_score_gain:+.4f}"
    )

    print(
        f"Average Relevant Score Gain: "
        f"{relevant_score_gain:+.4f}"
    )

    # =====================================================
    # Save JSON
    # =====================================================

    output = {
        "settings": {
            "final_k":
                FINAL_K,

            "top_per_query":
                TOP_PER_QUERY,

            "multi_query_count":
                MULTI_QUERY_COUNT,

            "embedding_model":
                embedding_model,

            "query_generator_model":
                chat_model,

            "fusion_method":
                "max_real_cosine_similarity"
        },

        "retrieval_metrics": {
            "original":
                original_summary,

            "single_rewrite":
                rewrite_summary,

            "multi_query":
                multi_summary
        },

        "similarity_summary": {
            "average_original_top_similarity":
                avg_original_top,

            "average_single_rewrite_top_similarity":
                avg_rewrite_top,

            "average_multi_query_top_similarity":
                avg_multi_top,

            "average_original_relevant_similarity":
                avg_original_relevant,

            "average_single_rewrite_relevant_similarity":
                avg_rewrite_relevant,

            "average_multi_query_relevant_similarity":
                avg_multi_relevant,

            "top_score_improved_questions":
                top_score_improved,

            "relevant_score_improved_questions":
                relevant_score_improved,

            "total_questions":
                total,

            "average_top_score_gain":
                top_score_gain,

            "average_relevant_score_gain":
                relevant_score_gain
        },

        "questions":
            detailed_results
    }

    with open(
        "multi_query_evaluation.json",
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
        "multi_query_evaluation.json"
    )
    print("=" * 70)


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    main()