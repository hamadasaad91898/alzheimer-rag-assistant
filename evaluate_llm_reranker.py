import os
import json
import math

from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client


load_dotenv(override=True)


# Config
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


# Clients
openai_client = OpenAI(
    api_key=azure_key,
    base_url=azure_endpoint.rstrip("/")
    + "/openai/v1/"
)

supabase = create_client(
    supabase_url,
    supabase_key
)


CANDIDATE_K = 10
FINAL_K = 5

EVAL_FILE = "eval_questions.json"
OUTPUT_FILE = "llm_reranker_evaluation.json"


RERANK_PROMPT = """
You are a strict passage reranker for a
Retrieval-Augmented Generation system.

Your only task is to rank the provided passages
according to how directly relevant they are to
the user's exact question.

Rules:

1. Rank passages by how directly they help answer
   the exact question.

2. A passage that directly explains the requested
   concept must rank above a passage that only
   mentions related terms.

3. Do not use outside knowledge.

4. Do not answer the question.

5. Do not explain your ranking.

6. Rank every provided Chunk ID exactly once.

7. Do not invent Chunk IDs.

8. Return valid JSON only.

Required format:

{
  "ranked_chunk_ids": [1, 2, 3, 4]
}
""".strip()


def create_embedding(text):
    response = openai_client.embeddings.create(
        model=embedding_model,
        input=text
    )

    return response.data[0].embedding


def retrieve_candidates(question):
    query_embedding = create_embedding(
        question
    )

    response = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": CANDIDATE_K
        }
    ).execute()

    return response.data or []


def build_rerank_input(
    question,
    candidates
):
    passages = []

    for index, chunk in enumerate(
        candidates,
        start=1
    ):
        pages = chunk.get(
            "pages"
        ) or []

        passages.append(
            f"""
PASSAGE {index}

Chunk ID: {chunk["chunk_id"]}
Section: {chunk["section"]}
Pages: {pages}

Content:
{chunk["content"]}
""".strip()
        )

    joined_passages = (
        "\n\n---\n\n".join(
            passages
        )
    )

    return f"""
Question:
{question}

Passages:
{joined_passages}
""".strip()


def parse_ranking(text):
    text = text.strip()

    try:
        data = json.loads(text)

    except json.JSONDecodeError:

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            raise ValueError(
                "LLM did not return valid JSON."
            )

        data = json.loads(
            text[start:end + 1]
        )

    ranked_ids = data.get(
        "ranked_chunk_ids"
    )

    if not isinstance(
        ranked_ids,
        list
    ):
        raise ValueError(
            "Missing ranked_chunk_ids."
        )

    return [
        int(chunk_id)
        for chunk_id in ranked_ids
    ]


def rerank_with_llm(
    question,
    candidates
):
    candidate_ids = [
        chunk["chunk_id"]
        for chunk in candidates
    ]

    user_input = build_rerank_input(
        question,
        candidates
    )

    response = (
        openai_client.responses.create(
            model=chat_model,
            instructions=RERANK_PROMPT,
            input=user_input,
            max_output_tokens=400
        )
    )

    ranked_ids = parse_ranking(
        response.output_text
    )

    cleaned_ids = []

    # Keep valid unique IDs only
    for chunk_id in ranked_ids:
        if (
            chunk_id in candidate_ids
            and chunk_id
            not in cleaned_ids
        ):
            cleaned_ids.append(
                chunk_id
            )

    # Add missing IDs using
    # original vector order
    for chunk_id in candidate_ids:
        if chunk_id not in cleaned_ids:
            cleaned_ids.append(
                chunk_id
            )

    return cleaned_ids


def calculate_dcg(
    retrieved,
    relevant
):
    dcg = 0.0

    for index, chunk_id in enumerate(
        retrieved
    ):
        if chunk_id in relevant:
            dcg += (
                1
                / math.log2(
                    index + 2
                )
            )

    return dcg


def calculate_ndcg(
    retrieved,
    relevant,
    k
):
    dcg = calculate_dcg(
        retrieved[:k],
        relevant
    )

    ideal_count = min(
        len(relevant),
        k
    )

    if ideal_count == 0:
        return 0.0

    ideal_dcg = sum(
        1
        / math.log2(
            index + 2
        )
        for index in range(
            ideal_count
        )
    )

    return dcg / ideal_dcg


with open(
    EVAL_FILE,
    "r",
    encoding="utf-8"
) as file:
    questions = json.load(file)


print("=" * 80)
print("GPT-5.6 SOL LLM RERANKER EVALUATION")
print("=" * 80)

print()
print(
    "Questions:",
    len(questions)
)

print(
    "Vector candidates:",
    CANDIDATE_K
)

print(
    "Final Top K:",
    FINAL_K
)


hit_scores = []
precision_scores = []
recall_scores = []
reciprocal_ranks = []
ndcg_scores = []

results_output = []


for question_index, item in enumerate(
    questions,
    start=1
):
    question = item["question"]

    relevant = set(
        item["relevant_chunk_ids"]
    )

    candidates = retrieve_candidates(
        question
    )

    vector_ids = [
        row["chunk_id"]
        for row in candidates
    ]

    print()
    print("=" * 80)

    print(
        f"QUESTION "
        f"{question_index}/"
        f"{len(questions)}"
    )

    print("=" * 80)

    print(question)

    try:
        llm_ranking = (
            rerank_with_llm(
                question,
                candidates
            )
        )

    except Exception as error:
        print(
            "Reranker error:",
            error
        )

        print(
            "Using vector order "
            "for this question."
        )

        llm_ranking = vector_ids


    retrieved = (
        llm_ranking[:FINAL_K]
    )

    found = [
        chunk_id
        for chunk_id in retrieved
        if chunk_id in relevant
    ]


    # Hit Rate
    hit = (
        1
        if found
        else 0
    )


    # Precision
    precision = (
        len(found) / FINAL_K
        if FINAL_K > 0
        else 0
    )


    # Recall
    recall = (
        len(found)
        / len(relevant)
        if relevant
        else 0
    )


    # Reciprocal Rank
    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(
        retrieved,
        start=1
    ):
        if chunk_id in relevant:
            reciprocal_rank = (
                1 / rank
            )

            break


    # nDCG
    ndcg = calculate_ndcg(
        retrieved,
        relevant,
        FINAL_K
    )


    hit_scores.append(
        hit
    )

    precision_scores.append(
        precision
    )

    recall_scores.append(
        recall
    )

    reciprocal_ranks.append(
        reciprocal_rank
    )

    ndcg_scores.append(
        ndcg
    )


    print(
        "Expected chunks:",
        sorted(relevant)
    )

    print(
        "Vector Top 10:",
        vector_ids
    )

    print(
        "LLM Ranked Top 10:",
        llm_ranking
    )

    print(
        "Final Top 5:",
        retrieved
    )

    print()

    print(
        "Hit@5:",
        hit
    )

    print(
        "Precision@5:",
        round(
            precision,
            3
        )
    )

    print(
        "Recall@5:",
        round(
            recall,
            3
        )
    )

    print(
        "Reciprocal Rank:",
        round(
            reciprocal_rank,
            3
        )
    )

    print(
        "nDCG@5:",
        round(
            ndcg,
            3
        )
    )

    print(
        "Found relevant chunks:",
        found
    )


    results_output.append(
        {
            "question": question,
            "expected_chunks": sorted(
                relevant
            ),
            "vector_top_10": vector_ids,
            "llm_ranked_top_10": llm_ranking,
            "final_top_5": retrieved,
            "found_relevant_chunks": found,
            "hit_at_5": hit,
            "precision_at_5": precision,
            "recall_at_5": recall,
            "reciprocal_rank": reciprocal_rank,
            "ndcg_at_5": ndcg
        }
    )


question_count = len(
    questions
)


hit_rate = (
    sum(hit_scores)
    / question_count
)

mean_precision = (
    sum(precision_scores)
    / question_count
)

mean_recall = (
    sum(recall_scores)
    / question_count
)

mrr = (
    sum(reciprocal_ranks)
    / question_count
)

mean_ndcg = (
    sum(ndcg_scores)
    / question_count
)


print()
print("=" * 80)
print("FINAL RESULTS")
print("=" * 80)

print(
    "Hit Rate@5:",
    round(
        hit_rate,
        4
    )
)

print(
    "Mean Precision@5:",
    round(
        mean_precision,
        4
    )
)

print(
    "Mean Recall@5:",
    round(
        mean_recall,
        4
    )
)

print(
    "MRR:",
    round(
        mrr,
        4
    )
)

print(
    "Mean nDCG@5:",
    round(
        mean_ndcg,
        4
    )
)


output = {
    "questions": question_count,
    "vector_candidate_k": CANDIDATE_K,
    "final_k": FINAL_K,
    "reranker": chat_model,
    "metrics": {
        "hit_rate_at_5": hit_rate,
        "mean_precision_at_5": mean_precision,
        "mean_recall_at_5": mean_recall,
        "mrr": mrr,
        "mean_ndcg_at_5": mean_ndcg
    },
    "results": results_output
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        output,
        file,
        indent=2,
        ensure_ascii=False
    )


print()
print(
    "Saved to:",
    OUTPUT_FILE
)

print("=" * 80)