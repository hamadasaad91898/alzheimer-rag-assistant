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
RERANK_MAX_OUTPUT_TOKENS = 1200

QUERY_REWRITE_MAX_OUTPUT_TOKENS = 500

MULTI_QUERY_MAX_ATTEMPTS = 2
MULTI_QUERY_MAX_OUTPUT_TOKENS = 800


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

11. Keep the rewritten query concise
    and standalone.

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
# Multi-Query Generator
# =========================================================

MULTI_QUERY_PROMPT = """
You generate alternative retrieval queries for an
Alzheimer's disease Retrieval-Augmented Generation system.

You are NOT answering the user's question.

You will receive:

1. The original user question.
2. A primary rewritten English retrieval query.

Generate exactly TWO alternative English retrieval queries.

The alternatives must represent the SAME information need.

Their purpose is only to improve semantic retrieval
against a scientific Alzheimer's disease document.

Rules:

1. Preserve the original intent exactly.

2. Do not broaden the user's question.

3. Do not remove important parts of the intent.

4. Do not add facts, assumptions, symptoms,
   treatments, medications, diagnoses,
   mechanisms, or recommendations that
   were not requested.

5. Use alternative scientific terminology
   when useful.

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

8. Keep both alternatives concise.

Return exactly:

ALT_1: ...
ALT_2: ...

No explanation.
No markdown.
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
        re.IGNORECASE
    )

    alt_2_match = re.search(
        r"ALT_2\s*:\s*(.+)",
        text,
        re.IGNORECASE
    )

    if (
        not alt_1_match
        or not alt_2_match
    ):
        raise ValueError(
            "Could not parse ALT_1 / ALT_2."
        )

    alt_1 = (
        alt_1_match.group(1).strip()
    )

    alt_2 = (
        alt_2_match.group(1).strip()
    )

    if not alt_1 or not alt_2:
        raise ValueError(
            "Alternative query is empty."
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

Return exactly two lines:

ALT_1: ...
ALT_2: ...

No other text.
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

            unique_queries = []

            for query in queries:
                query = query.strip()

                if not query:
                    continue

                normalized = query.lower()

                exists = any(
                    existing.lower()
                    == normalized
                    for existing in unique_queries
                )

                if not exists:
                    unique_queries.append(
                        query
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
        "Multi-query generation failed: "
        f"{last_error}"
    )


# =========================================================
# Multi-Query Retrieval
# =========================================================

def multi_query_search(queries):
    """
    Search each query independently.

    If the same chunk appears more than once,
    keep the highest REAL cosine similarity.

    No artificial score modification.
    """

    merged = {}

    for query_index, query in enumerate(
        queries,
        start=1
    ):
        chunks = vector_search(
            query,
            CANDIDATE_K
        )

        for chunk in chunks:
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
                new_chunk = dict(
                    chunk
                )

                new_chunk[
                    "chunk_id"
                ] = chunk_id

                new_chunk[
                    "similarity"
                ] = similarity

                new_chunk[
                    "best_query"
                ] = query

                new_chunk[
                    "best_query_index"
                ] = query_index

                new_chunk[
                    "query_scores"
                ] = {
                    str(query_index):
                        similarity
                }

                merged[
                    chunk_id
                ] = new_chunk

            else:
                merged[
                    chunk_id
                ][
                    "query_scores"
                ][
                    str(query_index)
                ] = similarity

                current_score = float(
                    merged[
                        chunk_id
                    ][
                        "similarity"
                    ]
                )

                if similarity > current_score:
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

    ranked_chunks = sorted(
        merged.values(),
        key=lambda chunk: float(
            chunk.get(
                "similarity",
                0
            )
        ),
        reverse=True
    )

    return ranked_chunks[
        :CANDIDATE_K
    ]


# =========================================================
# Stable GPT Reranker
# =========================================================

RERANK_PROMPT = """
You are a strict passage reranker for an
Alzheimer's disease Retrieval-Augmented
Generation system.

Your ONLY task is to rank candidate passages.

Do not answer the medical question.

Rank according to the ORIGINAL user question.

Ranking priority:

1. Direct relevance to the exact question.

2. Specific passages above broad passages.

3. Passages containing enough information to answer
   the question above passages that only mention
   related concepts.

4. Preserve the original user intent.

5. If two passages are equally relevant,
   prefer the one with higher Vector Similarity.

Rules:

- Use only the provided passages.
- Do not use outside knowledge.
- Do not explain the ranking.
- Do not invent Chunk IDs.
- Include every candidate Chunk ID exactly once.

OUTPUT:

Return ONLY Chunk IDs separated by commas.

Example:

12,13,8,10,9

No JSON.
No brackets.
No markdown.
No explanation.
""".strip()


def build_rerank_input(
    original_question,
    retrieval_query,
    candidates
):
    passages = []

    for index, chunk in enumerate(
        candidates,
        start=1
    ):
        similarity = float(
            chunk.get(
                "similarity",
                0
            )
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

    joined_passages = (
        "\n\n---\n\n"
    ).join(
        passages
    )

    return f"""
ORIGINAL USER QUESTION:

{original_question}


PRIMARY RETRIEVAL QUERY:

{retrieval_query}


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
            "No Chunk IDs found."
        )

    parsed_ids = []

    for number in numbers:
        chunk_id = int(
            number
        )

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
            f"Reranker missed IDs: "
            f"{missing_ids}"
        )

    if len(parsed_ids) != len(valid_ids):
        raise ValueError(
            "Invalid reranker ranking."
        )

    return parsed_ids


def rerank_chunks(
    original_question,
    retrieval_query,
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
        retrieval_query,
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

Return exactly these Chunk IDs,
each exactly once:

{",".join(str(x) for x in valid_ids)}

Return ONLY comma-separated Chunk IDs.
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

            return [
                chunk_map[chunk_id]
                for chunk_id
                in ranked_ids[:FINAL_K]
            ]

        except Exception as error:
            last_error = error

            print(
                f"Reranker attempt "
                f"{attempt} failed: "
                f"{error}"
            )

            print(
                "Raw output:",
                repr(raw_output)
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
                item["hit_rate"]
                for item in results
            ) / count,

        "precision_at_5":
            sum(
                item["precision"]
                for item in results
            ) / count,

        "recall_at_5":
            sum(
                item["recall"]
                for item in results
            ) / count,

        "mrr":
            sum(
                item["mrr"]
                for item in results
            ) / count,

        "ndcg_at_5":
            sum(
                item["ndcg"]
                for item in results
            ) / count
    }


# =========================================================
# Similarity Helpers
# =========================================================

def top_similarity(chunks):
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


def best_relevant_similarity(
    chunks,
    relevant_ids
):
    relevant = set(
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
        ) in relevant
    ]

    if not scores:
        return None

    return max(
        scores
    )


# =========================================================
# Print
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

    single_vector_results = []
    single_reranker_results = []

    multi_vector_results = []
    multi_reranker_results = []

    details = []

    single_reranker_failures = 0
    multi_reranker_failures = 0

    single_relevant_scores = []
    multi_relevant_scores = []

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
        # Query Rewrite
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

            rewritten_query = question

        print()
        print(
            f"Rewritten:"
        )

        print(
            rewritten_query
        )

        # =================================================
        # SINGLE QUERY RETRIEVAL
        # =================================================

        single_candidates = vector_search(
            rewritten_query,
            CANDIDATE_K
        )

        single_vector_top5 = (
            single_candidates[:FINAL_K]
        )

        single_vector_ids = [
            int(chunk["chunk_id"])
            for chunk
            in single_vector_top5
        ]

        single_vector_metrics = (
            evaluate_question(
                single_vector_ids,
                relevant_ids
            )
        )

        single_vector_results.append(
            single_vector_metrics
        )

        # =================================================
        # SINGLE QUERY RERANKER
        # =================================================

        single_fallback = False

        try:
            single_reranked = rerank_chunks(
                question,
                rewritten_query,
                single_candidates
            )

        except Exception as error:
            single_reranker_failures += 1
            single_fallback = True

            print(
                "Single-query reranker failed:",
                error
            )

            single_reranked = (
                single_candidates[:FINAL_K]
            )

        single_reranked_ids = [
            int(chunk["chunk_id"])
            for chunk
            in single_reranked
        ]

        single_reranker_metrics = (
            evaluate_question(
                single_reranked_ids,
                relevant_ids
            )
        )

        single_reranker_results.append(
            single_reranker_metrics
        )

        # =================================================
        # MULTI QUERY GENERATION
        # =================================================

        try:
            multi_queries = generate_multi_queries(
                question,
                rewritten_query
            )

        except Exception as error:
            print(
                "Multi-query error:",
                error
            )

            multi_queries = [
                rewritten_query
            ]

        # =================================================
        # MULTI QUERY RETRIEVAL
        # =================================================

        multi_candidates = multi_query_search(
            multi_queries
        )

        multi_vector_top5 = (
            multi_candidates[:FINAL_K]
        )

        multi_vector_ids = [
            int(chunk["chunk_id"])
            for chunk
            in multi_vector_top5
        ]

        multi_vector_metrics = (
            evaluate_question(
                multi_vector_ids,
                relevant_ids
            )
        )

        multi_vector_results.append(
            multi_vector_metrics
        )

        # =================================================
        # MULTI QUERY RERANKER
        # =================================================

        multi_fallback = False

        try:
            multi_reranked = rerank_chunks(
                question,
                rewritten_query,
                multi_candidates
            )

        except Exception as error:
            multi_reranker_failures += 1
            multi_fallback = True

            print(
                "Multi-query reranker failed:",
                error
            )

            multi_reranked = (
                multi_candidates[:FINAL_K]
            )

        multi_reranked_ids = [
            int(chunk["chunk_id"])
            for chunk
            in multi_reranked
        ]

        multi_reranker_metrics = (
            evaluate_question(
                multi_reranked_ids,
                relevant_ids
            )
        )

        multi_reranker_results.append(
            multi_reranker_metrics
        )

        # =================================================
        # Similarity
        # =================================================

        single_best_relevant = (
            best_relevant_similarity(
                single_candidates,
                relevant_ids
            )
        )

        multi_best_relevant = (
            best_relevant_similarity(
                multi_candidates,
                relevant_ids
            )
        )

        if single_best_relevant is not None:
            single_relevant_scores.append(
                single_best_relevant
            )

        if multi_best_relevant is not None:
            multi_relevant_scores.append(
                multi_best_relevant
            )

        # =================================================
        # Print Current Question
        # =================================================

        print()
        print(
            f"Relevant:              "
            f"{relevant_ids}"
        )

        print(
            f"Single Vector Top5:    "
            f"{single_vector_ids}"
        )

        print(
            f"Single + Reranker:     "
            f"{single_reranked_ids}"
        )

        print(
            f"Multi Vector Top5:     "
            f"{multi_vector_ids}"
        )

        print(
            f"Multi + Reranker:      "
            f"{multi_reranked_ids}"
        )

        print()

        print(
            f"Single Top Similarity: "
            f"{top_similarity(single_candidates):.4f}"
        )

        print(
            f"Multi Top Similarity:  "
            f"{top_similarity(multi_candidates):.4f}"
        )

        if single_best_relevant is not None:
            print(
                "Single Best Relevant:  "
                f"{single_best_relevant:.4f}"
            )

        if multi_best_relevant is not None:
            print(
                "Multi Best Relevant:   "
                f"{multi_best_relevant:.4f}"
            )

        print()

        print(
            f"Single Fallback: "
            f"{single_fallback}"
        )

        print(
            f"Multi Fallback:  "
            f"{multi_fallback}"
        )

        # =================================================
        # Save Detail
        # =================================================

        details.append(
            {
                "question":
                    question,

                "relevant_chunk_ids":
                    relevant_ids,

                "rewritten_query":
                    rewritten_query,

                "multi_queries":
                    multi_queries,

                "single_vector_ids":
                    single_vector_ids,

                "single_reranker_ids":
                    single_reranked_ids,

                "multi_vector_ids":
                    multi_vector_ids,

                "multi_reranker_ids":
                    multi_reranked_ids,

                "single_best_relevant_similarity":
                    single_best_relevant,

                "multi_best_relevant_similarity":
                    multi_best_relevant,

                "single_reranker_fallback":
                    single_fallback,

                "multi_reranker_fallback":
                    multi_fallback
            }
        )

    # =====================================================
    # Final Summaries
    # =====================================================

    single_vector_summary = (
        average_metrics(
            single_vector_results
        )
    )

    single_reranker_summary = (
        average_metrics(
            single_reranker_results
        )
    )

    multi_vector_summary = (
        average_metrics(
            multi_vector_results
        )
    )

    multi_reranker_summary = (
        average_metrics(
            multi_reranker_results
        )
    )

    print_metrics(
        "1. SINGLE QUERY + VECTOR",
        single_vector_summary
    )

    print_metrics(
        "2. SINGLE QUERY + VECTOR + GPT RERANKER",
        single_reranker_summary
    )

    print_metrics(
        "3. MULTI-QUERY + VECTOR",
        multi_vector_summary
    )

    print_metrics(
        "4. MULTI-QUERY + VECTOR + GPT RERANKER",
        multi_reranker_summary
    )

    # =====================================================
    # Similarity Summary
    # =====================================================

    average_single_relevant = (
        sum(single_relevant_scores)
        / len(single_relevant_scores)
        if single_relevant_scores
        else 0.0
    )

    average_multi_relevant = (
        sum(multi_relevant_scores)
        / len(multi_relevant_scores)
        if multi_relevant_scores
        else 0.0
    )

    similarity_gain = (
        average_multi_relevant
        - average_single_relevant
    )

    print()
    print("=" * 70)
    print("SIMILARITY COMPARISON")
    print("=" * 70)

    print(
        "Average Single Relevant Similarity: "
        f"{average_single_relevant:.4f}"
    )

    print(
        "Average Multi Relevant Similarity:  "
        f"{average_multi_relevant:.4f}"
    )

    print(
        "Average Similarity Gain:            "
        f"{similarity_gain:+.4f}"
    )

    # =====================================================
    # Reliability
    # =====================================================

    print()
    print("=" * 70)
    print("RERANKER RELIABILITY")
    print("=" * 70)

    print(
        f"Single Query Failures: "
        f"{single_reranker_failures}"
    )

    print(
        f"Multi Query Failures:  "
        f"{multi_reranker_failures}"
    )

    single_success_rate = (
        (total - single_reranker_failures)
        / total
    )

    multi_success_rate = (
        (total - multi_reranker_failures)
        / total
    )

    print(
        f"Single Success Rate:   "
        f"{single_success_rate:.4f}"
    )

    print(
        f"Multi Success Rate:    "
        f"{multi_success_rate:.4f}"
    )

    # =====================================================
    # Save
    # =====================================================

    output = {
        "settings": {
            "candidate_k":
                CANDIDATE_K,

            "final_k":
                FINAL_K,

            "embedding_model":
                embedding_model,

            "reranker_model":
                chat_model,

            "fusion_method":
                "max_real_cosine_similarity"
        },

        "single_query_vector":
            single_vector_summary,

        "single_query_reranker":
            single_reranker_summary,

        "multi_query_vector":
            multi_vector_summary,

        "multi_query_reranker":
            multi_reranker_summary,

        "similarity": {
            "average_single_relevant_similarity":
                average_single_relevant,

            "average_multi_relevant_similarity":
                average_multi_relevant,

            "average_gain":
                similarity_gain
        },

        "reliability": {
            "single_failures":
                single_reranker_failures,

            "multi_failures":
                multi_reranker_failures,

            "single_success_rate":
                single_success_rate,

            "multi_success_rate":
                multi_success_rate
        },

        "questions":
            details
    }

    with open(
        "multi_query_reranker_evaluation.json",
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
        "multi_query_reranker_evaluation.json"
    )

    print("=" * 70)


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    main()