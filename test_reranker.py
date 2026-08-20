import json

from flashrank import Ranker, RerankRequest


# Load current chunks
with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)


# Same Tau question we tested before
query = "What role does tau protein play in Alzheimer's disease?"


# Vector search results from our previous retrieval test
candidate_ids = [13, 12, 28, 9, 36]


# Get candidate chunks
chunk_map = {
    chunk["chunk_id"]: chunk
    for chunk in chunks
}

passages = []

for chunk_id in candidate_ids:
    chunk = chunk_map[chunk_id]

    passages.append(
        {
            "id": chunk["chunk_id"],
            "text": chunk["content"],
            "meta": {
                "section": chunk["section"],
                "pages": chunk["pages"]
            }
        }
    )


print("=" * 70)
print("BEFORE RERANKING")
print("=" * 70)

for rank, chunk_id in enumerate(candidate_ids, start=1):
    chunk = chunk_map[chunk_id]

    print(
        f"#{rank} | "
        f"Chunk {chunk_id} | "
        f"{chunk['section']}"
    )


# Load reranker
ranker = Ranker(
    model_name="ms-marco-MiniLM-L-12-v2",
    cache_dir="./flashrank_cache",
    max_length=512
)


# Rerank
request = RerankRequest(
    query=query,
    passages=passages
)

results = ranker.rerank(request)


print()
print("=" * 70)
print("AFTER RERANKING")
print("=" * 70)

for rank, result in enumerate(results, start=1):
    print(
        f"#{rank} | "
        f"Chunk {result['id']} | "
        f"Score: {result['score']:.4f} | "
        f"{result['meta']['section']}"
    )