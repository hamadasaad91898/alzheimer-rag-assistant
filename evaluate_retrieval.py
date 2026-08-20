import os
import json
import math

from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client


load_dotenv(override=True)


# Load config
azure_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")


# Create clients
openai_client = OpenAI(
    api_key=azure_key,
    base_url=azure_endpoint.rstrip("/") + "/openai/v1/"
)

supabase = create_client(
    supabase_url,
    supabase_key
)


TOP_K = 5
EVAL_FILE = "eval_questions.json"
OUTPUT_FILE = "retrieval_evaluation.json"


def create_embedding(text):
    response = openai_client.embeddings.create(
        model=embedding_model,
        input=text
    )

    return response.data[0].embedding


def calculate_dcg(retrieved, relevant):
    dcg = 0.0

    for index, chunk_id in enumerate(retrieved):
        if chunk_id in relevant:
            dcg += 1 / math.log2(index + 2)

    return dcg


def calculate_ndcg(retrieved, relevant, k):
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
        1 / math.log2(index + 2)
        for index in range(ideal_count)
    )

    return dcg / ideal_dcg


with open(
    EVAL_FILE,
    "r",
    encoding="utf-8"
) as file:
    questions = json.load(file)


print("=" * 80)
print("RETRIEVAL EVALUATION")
print("=" * 80)

print()
print("Questions:", len(questions))
print("Top K:", TOP_K)


results = []

hit_scores = []
precision_scores = []
recall_scores = []
reciprocal_ranks = []
ndcg_scores = []


for question_index, item in enumerate(
    questions,
    start=1
):
    question = item["question"]

    relevant = set(
        item["relevant_chunk_ids"]
    )


    query_embedding = create_embedding(
        question
    )


    response = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": TOP_K
        }
    ).execute()


    rows = response.data or []


    retrieved = [
        row["chunk_id"]
        for row in rows
    ]


    found = [
        chunk_id
        for chunk_id in retrieved
        if chunk_id in relevant
    ]


    # Hit Rate
    hit = 1 if found else 0


    # Precision
    precision = (
        len(found) / TOP_K
        if TOP_K > 0
        else 0
    )


    # Recall
    recall = (
        len(found) / len(relevant)
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
            reciprocal_rank = 1 / rank
            break


    # nDCG
    ndcg = calculate_ndcg(
        retrieved,
        relevant,
        TOP_K
    )


    hit_scores.append(hit)
    precision_scores.append(precision)
    recall_scores.append(recall)
    reciprocal_ranks.append(
        reciprocal_rank
    )
    ndcg_scores.append(ndcg)


    print()
    print("=" * 80)
    print(
        f"QUESTION "
        f"{question_index}/"
        f"{len(questions)}"
    )
    print("=" * 80)

    print(question)

    print(
        "Expected chunks:",
        sorted(relevant)
    )

    print(
        "Retrieved chunks:",
        retrieved
    )


    for rank, row in enumerate(
        rows,
        start=1
    ):
        print(
            f"  #{rank} "
            f"Chunk {row['chunk_id']} | "
            f"Similarity: "
            f"{row['similarity']:.4f}"
        )


    print()
    print("Hit@5:", hit)

    print(
        "Precision@5:",
        round(precision, 3)
    )

    print(
        "Recall@5:",
        round(recall, 3)
    )

    print(
        "Reciprocal Rank:",
        round(reciprocal_rank, 3)
    )

    print(
        "nDCG@5:",
        round(ndcg, 3)
    )

    print(
        "Found relevant chunks:",
        found
    )


    results.append(
        {
            "question": question,
            "expected_chunks": sorted(
                relevant
            ),
            "retrieved_chunks": retrieved,
            "found_relevant_chunks": found,
            "hit_at_5": hit,
            "precision_at_5": precision,
            "recall_at_5": recall,
            "reciprocal_rank": reciprocal_rank,
            "ndcg_at_5": ndcg
        }
    )


question_count = len(questions)


hit_rate = (
    sum(hit_scores) / question_count
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
    round(hit_rate, 4)
)

print(
    "Mean Precision@5:",
    round(mean_precision, 4)
)

print(
    "Mean Recall@5:",
    round(mean_recall, 4)
)

print(
    "MRR:",
    round(mrr, 4)
)

print(
    "Mean nDCG@5:",
    round(mean_ndcg, 4)
)


output = {
    "questions": question_count,
    "top_k": TOP_K,
    "metrics": {
        "hit_rate_at_5": hit_rate,
        "mean_precision_at_5": mean_precision,
        "mean_recall_at_5": mean_recall,
        "mrr": mrr,
        "mean_ndcg_at_5": mean_ndcg
    },
    "results": results
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