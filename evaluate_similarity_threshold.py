import os
import json
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

CANDIDATE_K = 10

QUERY_REWRITE_MAX_OUTPUT_TOKENS = 500

MULTI_QUERY_MAX_OUTPUT_TOKENS = 800
MULTI_QUERY_MAX_ATTEMPTS = 2


# =========================================================
# JSON Helper
# =========================================================

def parse_json_response(text):
    text = (
        text or ""
    ).strip()

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
# Multi-Query Generation
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
   mechanisms, risk factors, or recommendations
   that were not requested.

5. Use alternative scientific terminology
   when useful.

6. Do not change the disease or topic being asked about.

7. Do not transform an unrelated question into
   an Alzheimer's disease question.

8. Do not answer the question.

9. Keep both alternatives concise and standalone.

Return exactly:

ALT_1: ...
ALT_2: ...

No explanation.
No markdown.
No other text.
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

Return exactly:

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

                duplicate = any(
                    existing.lower() == normalized
                    for existing in unique_queries
                )

                if not duplicate:
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
# Embedding + Vector Search
# =========================================================

def create_embedding(text):
    response = openai_client.embeddings.create(
        model=embedding_model,
        input=text
    )

    return response.data[0].embedding


def vector_search(
    query,
    top_k=CANDIDATE_K
):
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
# Multi-Query Search
# =========================================================

def multi_query_search(queries):
    """
    Keep the highest REAL cosine similarity
    obtained for each chunk.

    No artificial score adjustment.
    """

    merged = {}

    for query_index, query in enumerate(
        queries,
        start=1
    ):
        results = vector_search(
            query,
            CANDIDATE_K
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
                item = dict(chunk)

                item["chunk_id"] = chunk_id
                item["similarity"] = similarity
                item["best_query"] = query
                item["best_query_index"] = query_index

                merged[chunk_id] = item

            else:
                current_score = float(
                    merged[
                        chunk_id
                    ]["similarity"]
                )

                if similarity > current_score:
                    merged[
                        chunk_id
                    ]["similarity"] = similarity

                    merged[
                        chunk_id
                    ]["best_query"] = query

                    merged[
                        chunk_id
                    ]["best_query_index"] = query_index

    ranked = sorted(
        merged.values(),
        key=lambda chunk: float(
            chunk.get(
                "similarity",
                0
            )
        ),
        reverse=True
    )

    return ranked[
        :CANDIDATE_K
    ]


# =========================================================
# Production-Like Score Pipeline
# =========================================================

def get_question_score(question):
    """
    Same basic retrieval logic as rag_chat.py:

    Original question
        -> rewrite
        -> 3 retrieval queries
        -> vector searches
        -> max real cosine similarity

    Returns the BEST vector similarity
    BEFORE reranking.
    """

    # Query rewriting
    try:
        rewritten_query = rewrite_query(
            question
        )

    except Exception as error:
        print(
            "Rewrite failed:",
            error
        )

        rewritten_query = question

    # Multi-query generation
    try:
        queries = generate_multi_queries(
            question,
            rewritten_query
        )

    except Exception as error:
        print(
            "Multi-query failed:",
            error
        )

        queries = [
            rewritten_query
        ]

    # Retrieval
    chunks = multi_query_search(
        queries
    )

    if not chunks:
        return {
            "question": question,
            "rewritten_query": rewritten_query,
            "queries": queries,
            "score": 0.0,
            "top_chunk_id": None,
            "top_section": None,
            "best_query": None
        }

    top_chunk = chunks[0]

    return {
        "question":
            question,

        "rewritten_query":
            rewritten_query,

        "queries":
            queries,

        "score":
            float(
                top_chunk.get(
                    "similarity",
                    0
                )
            ),

        "top_chunk_id":
            int(
                top_chunk[
                    "chunk_id"
                ]
            ),

        "top_section":
            top_chunk.get(
                "section"
            ),

        "best_query":
            top_chunk.get(
                "best_query"
            )
    }


# =========================================================
# Out-of-Scope Questions
# =========================================================

OUT_OF_SCOPE_QUESTIONS = [
    "What is agentic AI?",
    "How does a transformer language model work?",
    "What is quantum computing?",
    "How does blockchain technology work?",
    "What is the capital of France?",
    "How do solar panels generate electricity?",
    "What causes earthquakes?",
    "How does photosynthesis work?",
    "What is the history of the Roman Empire?",
    "How do I learn Python programming?",
    "What causes Parkinson's disease?",
    "How is Parkinson's disease diagnosed?",
    "What treatments are used for Parkinson's disease?",
    "What are the symptoms of multiple sclerosis?",
    "How is type 2 diabetes treated?",
    "What causes lung cancer?",
    "What are the treatments for breast cancer?",
    "How is hypertension diagnosed?",
    "ما هو الذكاء الاصطناعي؟",
    "ما هي أعراض مرض باركنسون؟"
]


# =========================================================
# Threshold Metrics
# =========================================================

def calculate_threshold_metrics(
    positive_scores,
    negative_scores,
    threshold
):
    # In-scope = positive
    # Out-of-scope = negative

    tp = sum(
        score >= threshold
        for score in positive_scores
    )

    fn = sum(
        score < threshold
        for score in positive_scores
    )

    fp = sum(
        score >= threshold
        for score in negative_scores
    )

    tn = sum(
        score < threshold
        for score in negative_scores
    )

    total = (
        tp + tn + fp + fn
    )

    accuracy = (
        (tp + tn) / total
        if total
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    balanced_accuracy = (
        recall + specificity
    ) / 2

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn)
        else 0.0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp)
        else 0.0
    )

    return {
        "threshold": threshold,

        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,

        "false_positive_rate":
            false_positive_rate,

        "false_negative_rate":
            false_negative_rate
    }


# =========================================================
# Threshold Selection
# =========================================================

def find_best_threshold(results):
    """
    Primary objective:
    maximize Balanced Accuracy.

    Tie breakers:
    1. higher specificity
    2. higher recall
    3. higher threshold
    """

    return max(
        results,
        key=lambda item: (
            item[
                "balanced_accuracy"
            ],
            item[
                "specificity"
            ],
            item[
                "recall"
            ],
            item[
                "threshold"
            ]
        )
    )


def find_safe_threshold(results):
    """
    Prefer a threshold with:

    Recall >= 95%
    Specificity >= 95%

    If multiple thresholds satisfy this,
    choose the LOWEST one to preserve
    as much in-scope recall as possible.
    """

    valid = [
        item
        for item in results
        if (
            item["recall"] >= 0.95
            and item["specificity"] >= 0.95
        )
    ]

    if not valid:
        return None

    return min(
        valid,
        key=lambda item:
            item["threshold"]
    )


# =========================================================
# Helpers
# =========================================================

def average(values):
    if not values:
        return 0.0

    return (
        sum(values)
        / len(values)
    )


def print_score_group(
    title,
    results
):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    for index, item in enumerate(
        results,
        start=1
    ):
        print(
            f"{index:02d}. "
            f"Score: {item['score']:.4f} | "
            f"Chunk: {item['top_chunk_id']} | "
            f"{item['question']}"
        )


# =========================================================
# Main
# =========================================================

def main():

    # =====================================================
    # Load Positive / In-Scope Questions
    # =====================================================

    with open(
        "eval_questions.json",
        "r",
        encoding="utf-8"
    ) as file:
        eval_questions = json.load(
            file
        )

    in_scope_questions = [
        item["question"]
        for item in eval_questions
    ]

    print()
    print("=" * 70)
    print("SIMILARITY THRESHOLD CALIBRATION")
    print("=" * 70)

    print(
        f"In-Scope Questions:     "
        f"{len(in_scope_questions)}"
    )

    print(
        f"Out-of-Scope Questions: "
        f"{len(OUT_OF_SCOPE_QUESTIONS)}"
    )

    # =====================================================
    # In-Scope Scores
    # =====================================================

    in_scope_results = []

    print()
    print("=" * 70)
    print("SCORING IN-SCOPE QUESTIONS")
    print("=" * 70)

    for index, question in enumerate(
        in_scope_questions,
        start=1
    ):
        print()
        print(
            f"[IN {index}/"
            f"{len(in_scope_questions)}]"
        )

        print(
            question
        )

        result = get_question_score(
            question
        )

        in_scope_results.append(
            result
        )

        print(
            f"Best Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Top Chunk: "
            f"{result['top_chunk_id']}"
        )

    # =====================================================
    # Out-of-Scope Scores
    # =====================================================

    out_scope_results = []

    print()
    print("=" * 70)
    print("SCORING OUT-OF-SCOPE QUESTIONS")
    print("=" * 70)

    for index, question in enumerate(
        OUT_OF_SCOPE_QUESTIONS,
        start=1
    ):
        print()
        print(
            f"[OUT {index}/"
            f"{len(OUT_OF_SCOPE_QUESTIONS)}]"
        )

        print(
            question
        )

        result = get_question_score(
            question
        )

        out_scope_results.append(
            result
        )

        print(
            f"Best Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Top Chunk: "
            f"{result['top_chunk_id']}"
        )

    # =====================================================
    # Score Arrays
    # =====================================================

    positive_scores = [
        item["score"]
        for item in in_scope_results
    ]

    negative_scores = [
        item["score"]
        for item in out_scope_results
    ]

    # =====================================================
    # Distribution Summary
    # =====================================================

    min_positive = min(
        positive_scores
    )

    max_negative = max(
        negative_scores
    )

    avg_positive = average(
        positive_scores
    )

    avg_negative = average(
        negative_scores
    )

    print_score_group(
        "IN-SCOPE SCORES",
        in_scope_results
    )

    print_score_group(
        "OUT-OF-SCOPE SCORES",
        out_scope_results
    )

    print()
    print("=" * 70)
    print("SCORE DISTRIBUTION")
    print("=" * 70)

    print(
        f"Average In-Scope Score:     "
        f"{avg_positive:.4f}"
    )

    print(
        f"Lowest In-Scope Score:      "
        f"{min_positive:.4f}"
    )

    print()

    print(
        f"Average Out-of-Scope Score: "
        f"{avg_negative:.4f}"
    )

    print(
        f"Highest Out-of-Scope Score: "
        f"{max_negative:.4f}"
    )

    print()

    separation_gap = (
        min_positive
        - max_negative
    )

    print(
        f"Separation Gap:             "
        f"{separation_gap:+.4f}"
    )

    # =====================================================
    # Threshold Sweep
    # =====================================================

    threshold_results = []

    threshold = 0.20

    while threshold <= 0.70:
        threshold = round(
            threshold,
            2
        )

        metrics = (
            calculate_threshold_metrics(
                positive_scores,
                negative_scores,
                threshold
            )
        )

        threshold_results.append(
            metrics
        )

        threshold += 0.01

    # =====================================================
    # Best Threshold
    # =====================================================

    best_threshold = (
        find_best_threshold(
            threshold_results
        )
    )

    safe_threshold = (
        find_safe_threshold(
            threshold_results
        )
    )

    # =====================================================
    # Print Useful Threshold Range
    # =====================================================

    print()
    print("=" * 70)
    print("THRESHOLD SWEEP")
    print("=" * 70)

    print(
        "Threshold | Acc    | "
        "Recall | Spec   | "
        "F1     | FP | FN"
    )

    print("-" * 70)

    for item in threshold_results:
        threshold_value = (
            item["threshold"]
        )

        # Print useful region only
        if 0.25 <= threshold_value <= 0.55:
            print(
                f"{threshold_value:9.2f} | "
                f"{item['accuracy']:.4f} | "
                f"{item['recall']:.4f} | "
                f"{item['specificity']:.4f} | "
                f"{item['f1']:.4f} | "
                f"{item['fp']:2d} | "
                f"{item['fn']:2d}"
            )

    # =====================================================
    # Final Recommendation
    # =====================================================

    print()
    print("=" * 70)
    print("BEST BALANCED THRESHOLD")
    print("=" * 70)

    print(
        f"Threshold:          "
        f"{best_threshold['threshold']:.2f}"
    )

    print(
        f"Accuracy:           "
        f"{best_threshold['accuracy']:.4f}"
    )

    print(
        f"Precision:          "
        f"{best_threshold['precision']:.4f}"
    )

    print(
        f"Recall:             "
        f"{best_threshold['recall']:.4f}"
    )

    print(
        f"Specificity:        "
        f"{best_threshold['specificity']:.4f}"
    )

    print(
        f"F1:                 "
        f"{best_threshold['f1']:.4f}"
    )

    print(
        f"Balanced Accuracy:  "
        f"{best_threshold['balanced_accuracy']:.4f}"
    )

    print(
        f"False Positives:    "
        f"{best_threshold['fp']}"
    )

    print(
        f"False Negatives:    "
        f"{best_threshold['fn']}"
    )

    print()
    print("=" * 70)
    print("SAFE THRESHOLD CANDIDATE")
    print("=" * 70)

    if safe_threshold is not None:
        print(
            f"Threshold:          "
            f"{safe_threshold['threshold']:.2f}"
        )

        print(
            f"Recall:             "
            f"{safe_threshold['recall']:.4f}"
        )

        print(
            f"Specificity:        "
            f"{safe_threshold['specificity']:.4f}"
        )

        print(
            f"False Positives:    "
            f"{safe_threshold['fp']}"
        )

        print(
            f"False Negatives:    "
            f"{safe_threshold['fn']}"
        )

    else:
        print(
            "No threshold achieved both "
            "Recall >= 0.95 and "
            "Specificity >= 0.95."
        )

    # =====================================================
    # Current Threshold Comparison
    # =====================================================

    current_threshold = 0.35

    current_metrics = (
        calculate_threshold_metrics(
            positive_scores,
            negative_scores,
            current_threshold
        )
    )

    print()
    print("=" * 70)
    print("CURRENT THRESHOLD = 0.35")
    print("=" * 70)

    print(
        f"Accuracy:        "
        f"{current_metrics['accuracy']:.4f}"
    )

    print(
        f"Recall:          "
        f"{current_metrics['recall']:.4f}"
    )

    print(
        f"Specificity:     "
        f"{current_metrics['specificity']:.4f}"
    )

    print(
        f"False Positives: "
        f"{current_metrics['fp']}"
    )

    print(
        f"False Negatives: "
        f"{current_metrics['fn']}"
    )

    # =====================================================
    # Save Results
    # =====================================================

    output = {
        "settings": {
            "candidate_k":
                CANDIDATE_K,

            "embedding_model":
                embedding_model,

            "query_generator_model":
                chat_model,

            "retrieval_method":
                "multi_query_max_real_cosine_similarity",

            "in_scope_count":
                len(
                    in_scope_questions
                ),

            "out_of_scope_count":
                len(
                    OUT_OF_SCOPE_QUESTIONS
                )
        },

        "distribution": {
            "average_in_scope":
                avg_positive,

            "minimum_in_scope":
                min_positive,

            "average_out_of_scope":
                avg_negative,

            "maximum_out_of_scope":
                max_negative,

            "separation_gap":
                separation_gap
        },

        "best_balanced_threshold":
            best_threshold,

        "safe_threshold_candidate":
            safe_threshold,

        "current_threshold_035":
            current_metrics,

        "threshold_sweep":
            threshold_results,

        "in_scope_results":
            in_scope_results,

        "out_of_scope_results":
            out_scope_results
    }

    with open(
        "similarity_threshold_evaluation.json",
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
        "similarity_threshold_evaluation.json"
    )
    print("=" * 70)


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    main()