import os
import json
import math

from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client
from flashrank import Ranker, RerankRequest


load_dotenv(override=True)


# Config
azure_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")


# Clients
openai_client = OpenAI(
    api_key=azure_key,
    base_url=azure_endpoint.rstrip("/") + "/openai/v1/"
)

supabase = create_client(
    supabase_url,
    supabase_key
)


# Reranker
ranker = Ranker(
    model_name="ms-marco-MiniLM-L-12-v2",
    cache_dir="./flashrank_cache",
    max_length=512
)


CANDIDATE_K = 10
FINAL_K = 5

EVAL_FILE = "eval_questions.json"
OUTPUT_FILE = "reranker_evaluation.json"


def create_embedding(text):
    response = openai_client.embeddings.create(
        model=embedding_model,
        input=text
    )

    return response.data[0].embedding


def retrieve_candidates(question):
    query_embedding = create_embedding(question)

    response = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": CANDIDATE_K
        }
    ).execute()

    return response.data or []


def rerank_chunks(question, candidates):
    passages = []

    for chunk in candidates:
        passages.append(
            {
                "id": chunk["chunk_id"],
                "text": chunk["content"],
                "meta": {
                    "section": chunk["section"],
                    "pages": chunk["pages"],
                    "source": chunk["source"],
                    "vector_similarity": chunk["similarity"]
                }
            }
        )

    request = RerankRequest(
        query=question,
        passages=passages
    )

    results = ranker.rerank(request)

    return results[:FINAL_K]


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
print("RERANKER EVALUATION")
print("=" * 80)

print()
print("Questions:", len(questions))
print("Vector candidates:", CANDIDATE_K)
print("Final Top K:", FINAL_K)


results_output = []

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

    # Vector retrieval
    candidates = retrieve_candidates(
        question
    )

    vector_ids = [
        row["chunk_id"]
        for row in candidates
    ]

    # Reranking
    reranked = rerank_chunks(
        question,
        candidates
    )

    retrieved = [
        result["id"]
        for result in reranked
    ]

    found = [
        chunk_id
        for chunk_id in retrieved
        if chunk_id in relevant
    ]


    # Metrics
    hit = 1 if found else 0

    precision = (
        len(found) / FINAL_K
        if FINAL_K > 0
        else 0
    )

    recall = (
        len(found) / len(relevant)
        if relevant
        else 0
    )

    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(
        retrieved,
        start=1
    ):
        if chunk_id in relevant:
            reciprocal_rank = 1 / rank
            break

    ndcg = calculate_ndcg(
        retrieved,
        relevant,
        FINAL_K
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
        "Vector Top 10:",
        vector_ids
    )

    print(
        "Reranked Top 5:",
        retrieved
    )


    for rank, result in enumerate(
        reranked,
        start=1
    ):
        print(
            f"  #{rank} "
            f"Chunk {result['id']} | "
            f"Rerank Score: "
            f"{result['score']:.4f}"
        )


    print()

    print(
        "Hit@5:",
        hit
    )

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


    results_output.append(
        {
            "question": question,
            "expected_chunks": sorted(
                relevant
            ),
            "vector_top_10": vector_ids,
            "reranked_top_5": retrieved,
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
    "vector_candidate_k": CANDIDATE_K,
    "final_k": FINAL_K,
    "reranker": "ms-marco-MiniLM-L-12-v2",
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