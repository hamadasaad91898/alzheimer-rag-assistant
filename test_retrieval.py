import os

from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client


load_dotenv()


azure_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
embedding_model = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
)

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")


client = OpenAI(
    api_key=azure_key,
    base_url=(
        azure_endpoint.rstrip("/")
        + "/openai/v1/"
    )
)


supabase = create_client(
    supabase_url,
    supabase_key
)


questions = [
    "What are the main risk factors for Alzheimer's disease?",

    "What are the typical clinical symptoms of Alzheimer's disease?",

    "What role does amyloid-beta play in Alzheimer's disease?",

    "What role does tau protein play in Alzheimer's disease?",

    "How does neuroinflammation contribute to Alzheimer's disease?",

    "What genetic factors are associated with Alzheimer's disease?",

    "How does the APOE epsilon 4 allele affect Alzheimer's disease risk?",

    "How is Alzheimer's disease diagnosed?",

    "What biomarkers are used for Alzheimer's disease diagnosis?",

    "What is the role of amyloid PET imaging in Alzheimer's disease?",

    "What is the role of tau PET imaging in Alzheimer's disease?",

    "What MRI findings are associated with Alzheimer's disease?",

    "What blood-based biomarkers are used for Alzheimer's disease?",

    "What is the importance of plasma p-tau217 in Alzheimer's disease diagnosis?",

    "How do cholinesterase inhibitors treat Alzheimer's disease?",

    "What is the role of memantine in Alzheimer's disease treatment?",

    "How does lecanemab work in Alzheimer's disease?",

    "How does donanemab work in Alzheimer's disease?",

    "What lifestyle interventions may reduce Alzheimer's disease risk?",

    "What are the main challenges and future directions in Alzheimer's disease treatment?"
]


for q_index, question in enumerate(
    questions,
    start=1
):
    response = client.embeddings.create(
        model=embedding_model,
        input=question
    )

    query_embedding = response.data[0].embedding


    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": 5
        }
    ).execute()


    rows = result.data or []


    print()
    print("=" * 80)
    print(f"QUESTION {q_index}/20")
    print(question)
    print("=" * 80)


    for rank, row in enumerate(
        rows,
        start=1
    ):
        print(
            f"{rank}. "
            f"Chunk {row['chunk_id']} | "
            f"{row['section']} | "
            f"Pages {row['pages']} | "
            f"Similarity "
            f"{row['similarity']:.4f}"
        )


print()
print("=" * 80)
print("20-question retrieval test completed")
print("=" * 80)