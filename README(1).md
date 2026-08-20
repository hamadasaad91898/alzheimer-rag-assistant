# 🧠 Alzheimer's Evidence-Grounded RAG Assistant

An evidence-grounded Retrieval-Augmented Generation (RAG) system for answering educational questions about Alzheimer's disease using a verified medical source.

The project combines semantic retrieval, multi-query search, reranking, evidence sufficiency validation, claim-level citation verification, medical safety classification, incremental indexing, and a Streamlit user interface.

> This system is designed for educational and informational use only. It does not provide patient-specific diagnosis, treatment decisions, or medication dosing.

---

## 🚀 Overview

Traditional RAG systems may generate answers when retrieved passages are only loosely related to the user's question.

This project takes a stricter approach.

Before an answer reaches the user, the system checks:

- Is the request medically safe?
- Was the user's intent preserved?
- Which document chunks are most relevant?
- Do the retrieved passages actually contain enough evidence to answer?
- Is every generated factual claim supported by its cited source?
- Are all citations valid?
- Should unsupported claims be removed?

The result is a RAG pipeline designed to reduce hallucination and produce traceable, evidence-grounded answers.

---

## ✨ Key Features

### 🔐 Medical Safety Layer

Every request is classified before retrieval.

The safety classifier distinguishes between:

- Educational questions
- Patient-specific diagnosis requests
- Patient-specific treatment requests
- Patient-specific medication dosage requests
- Medical emergencies
- Attempts to override clinician advice

Unsafe requests are blocked before retrieval or generation.

---

### 🔎 Query Rewriting

Arabic, informal Arabic, and English questions are converted into clear retrieval queries while preserving the original intent.

Example:

```text
User:
اي هو الزهايمر؟

Retrieval Query:
What is Alzheimer's disease?
```

The system does not use the rewritten query as the final answer input. It is used only to improve retrieval.

---

### 🔀 Multi-Query Retrieval

Instead of relying on a single search query, the system generates multiple semantically equivalent retrieval queries.

```text
Original Question
        ↓
Primary Rewrite
        ↓
Alternative Query 1
Alternative Query 2
        ↓
Vector Search
```

Results are merged and deduplicated using the chunk ID while preserving the highest real vector similarity score.

---

### 🧬 Vector Search

Embeddings are generated using:

```text
text-embedding-3-large
```

The vector database is powered by:

```text
Supabase + pgvector
```

The current embedding dimension is:

```text
3072
```

Vector similarity is used for:

- Retrieving relevant chunks
- Ranking candidates
- Debugging
- Confidence reporting

Similarity alone does **not** determine whether the system is allowed to answer.

---

### 🏆 LLM Reranking

The initial vector candidates are reranked based on their direct relevance to the original user question.

Pipeline:

```text
Vector Top 10
      ↓
LLM Reranker
      ↓
Top 5 Passages
```

The reranker prefers passages that directly answer the exact information need instead of passages that are only broadly related.

---

## 🧑‍⚖️ Evidence Sufficiency Judge

A dedicated LLM-based evidence judge analyzes the actual content of the reranked passages.

It answers only:

```text
SUFFICIENT
```

or:

```text
INSUFFICIENT
```

This solves an important RAG problem:

A relevant passage may have a moderate vector similarity score even though it clearly contains the answer.

Instead of refusing purely because of a fixed similarity threshold, the system checks whether the retrieved evidence itself is sufficient.

```text
Top 5 Passages
      ↓
Evidence Sufficiency Judge
      ↓
 ┌───────────────┐
 │               │
SUFFICIENT   INSUFFICIENT
 │               │
 ↓               ↓
Generate        Refuse
```

The judge is instructed to use only the supplied passages and not outside knowledge.

---

## 🧩 Atomic Claim Generation

The answer is not generated as one unrestricted paragraph.

Instead, the model produces atomic factual claims.

Example:

```text
CLAIM_1: Alzheimer's disease is a progressive neurodegenerative disorder.
CITES_1: 3

CLAIM_2: Age is an important risk factor.
CITES_2: 4
```

Each claim must reference one or more valid retrieved chunk IDs.

---

## ✅ Citation Validation

Citation IDs are validated programmatically.

The system checks that:

- A citation exists
- The cited chunk was actually retrieved
- The chunk ID is valid
- No fabricated citation was generated

Invalid citations automatically prevent the claim from reaching the final answer.

---

## 🔬 Claim-Level Evidence Verification

Every generated claim is independently verified against only its cited passages.

The Evidence Support Judge returns:

```text
SUPPORTED
```

or:

```text
UNSUPPORTED
```

A claim is included in the final answer only when:

```text
Citation Exists
      +
Citation Is Valid
      +
Evidence Judge = SUPPORTED
      ↓
Final Answer
```

Unsupported claims are removed automatically.

---

## 📚 Verified Citations

The final answer includes deterministic citation metadata such as:

```text
Section
Pages
Chunk ID
Source
Retrieval Score
```

Example:

```text
Section: 4.1 Typical presentation
Pages: 3, 4
Chunk ID: 5
Source: neurosci.pdf
Retrieval Score: 0.7430
```

This makes the answer traceable back to the original source document.

---

## 🔄 Incremental / Delta Indexing

The project includes an incremental indexing pipeline.

Instead of reprocessing and re-embedding the entire document whenever the knowledge base changes, the pipeline compares hashes.

### Unchanged Document

```text
PDF Hash Unchanged
      ↓
Skip Docling
Skip Cleaning
Skip Chunking
Skip Embeddings
Skip Database Changes
```

### Updated PDF with Unchanged Extracted Content

```text
PDF Hash Changed
      ↓
Docling Runs
      ↓
Chunks Compared
      ↓
Existing Embeddings Reused
```

### Changed Chunk

```text
Only Changed Chunk
      ↓
Generate New Embedding
      ↓
Update Supabase
```

This reduces unnecessary processing, API usage, and indexing cost.

---

## 🏗️ System Architecture

```text
                           USER QUESTION
                                 │
                                 ▼
                       ┌───────────────────┐
                       │ Safety Classifier │
                       └─────────┬─────────┘
                                 │
                     Unsafe ─────┴───── Safe
                       │                 │
                       ▼                 ▼
                    Refuse         Query Rewrite
                                         │
                                         ▼
                                  Multi-Query
                                         │
                           ┌─────────────┼─────────────┐
                           ▼             ▼             ▼
                          Q1            Q2            Q3
                           │             │             │
                           └─────────────┼─────────────┘
                                         ▼
                                Vector Retrieval
                                  Supabase / pgvector
                                         │
                                         ▼
                                      Top 10
                                         │
                                         ▼
                                   LLM Reranker
                                         │
                                         ▼
                                      Top 5
                                         │
                                         ▼
                           Evidence Sufficiency Judge
                                   │             │
                            INSUFFICIENT     SUFFICIENT
                                   │             │
                                   ▼             ▼
                                Refuse      Claim Generation
                                                  │
                                                  ▼
                                         Citation Validation
                                                  │
                                                  ▼
                                       Claim Evidence Judge
                                                  │
                                                  ▼
                                      Remove Unsupported Claims
                                                  │
                                                  ▼
                                      Deterministic Citations
                                                  │
                                                  ▼
                                           FINAL ANSWER
```

---

## 💬 Conversation Memory

The Streamlit interface supports temporary conversational context.

Example:

```text
User:
ما هي أعراض الزهايمر؟

User:
طب وعلاجه ايه؟
```

The second question can be resolved into a standalone question using recent conversation context.

Important:

- Memory is session-based
- It is not stored permanently
- Starting a new chat clears the conversation memory
- Medical safety intent must remain preserved during memory resolution

---

## 🖥️ Streamlit Interface

The project includes an interactive Streamlit application with:

- Chat interface
- Arabic and English support
- Dark mode
- Light mode
- Conversation memory
- Confidence indicators
- Citation coverage
- Verified claim count
- Verification rate
- Verified source viewer
- Similarity scores
- Safety messages
- Read-only administrator-managed knowledge base

Users cannot upload or modify source documents.

---

## 📄 Knowledge Source

The current knowledge base is built from the scientific review:

**Alzheimer's disease: A comprehensive review of epidemiology, pathophysiology, diagnosis, and treatment**

The source covers topics including:

- Epidemiology
- Risk factors
- Clinical presentation
- Pathophysiology
- Amyloid and tau hypotheses
- Diagnosis
- Differential diagnosis
- Treatment
- Prevention
- Conclusions

The knowledge base is managed by the project administrator.

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| UI | Streamlit |
| LLM | Azure OpenAI |
| Embeddings | text-embedding-3-large |
| Vector Database | Supabase / pgvector |
| Document Processing | Docling |
| Retrieval | Multi-query semantic search |
| Reranking | LLM-based reranking |
| Evidence Validation | LLM-as-a-Judge |
| Citation Validation | Deterministic Python validation |
| Deployment | Streamlit Community Cloud |
| Version Control | Git / GitHub |

---

## 📁 Project Structure

```text
alzheimer-rag-assistant/
│
├── rag_chat.py
├── streamlit_app.py
│
├── index_pipeline.py
├── incremental_ingest.py
│
├── docling.py
├── clean_docling_json.py
├── chunk_docling_json.py
│
├── test_embedding_model.py
├── test_retrieval.py
├── test_reranker.py
├── test_safety_classifier.py
├── test_incremental_pipeline.py
│
├── evaluate_retrieval.py
├── evaluate_query_rewriting.py
├── evaluate_multi_query.py
├── evaluate_reranker.py
├── evaluate_llm_reranker.py
├── evaluate_multi_query_reranker.py
├── evaluate_citation_coverage.py
├── evaluate_generation_quality.py
├── evaluate_similarity_threshold.py
├── evaluate_final_end_to_end.py
│
├── chunks.json
├── cleaned_docling.json
├── docling_output.json
├── docling_output.md
│
├── eval_questions.json
├── *_evaluation.json
│
├── FINAL_EVALUATION_REPORT.md
│
├── neurosci.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Environment Variables

Create a local `.env` file:

```env
AZURE_OPENAI_API_KEY=YOUR_KEY
AZURE_OPENAI_ENDPOINT=YOUR_ENDPOINT

AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
AZURE_CHAT_DEPLOYMENT=YOUR_CHAT_MODEL

SUPABASE_URL=YOUR_SUPABASE_URL
SUPABASE_KEY=YOUR_SUPABASE_KEY
```

> Never commit `.env`, API keys, database secrets, or Streamlit secrets to GitHub.

---

## 📦 Installation

```bash
git clone https://github.com/hamadasaad91898/alzheimer-rag-assistant.git
cd alzheimer-rag-assistant
pip install -r requirements.txt
```

---

## ▶️ Run the RAG in CLI Mode

```bash
python rag_chat.py
```

---

## 🌐 Run the Streamlit Interface

```bash
python -m streamlit run streamlit_app.py
```

---

## ☁️ Streamlit Community Cloud Deployment

Main application file:

```text
streamlit_app.py
```

Add credentials through:

```text
Streamlit Cloud
→ App Settings
→ Secrets
```

Example:

```toml
AZURE_OPENAI_API_KEY = "YOUR_KEY"
AZURE_OPENAI_ENDPOINT = "YOUR_ENDPOINT"

AZURE_OPENAI_EMBEDDING_DEPLOYMENT = "text-embedding-3-large"
AZURE_CHAT_DEPLOYMENT = "YOUR_CHAT_MODEL"

SUPABASE_URL = "YOUR_SUPABASE_URL"
SUPABASE_KEY = "YOUR_SUPABASE_KEY"
```

Do not upload production credentials to GitHub.

---

## 🧪 Evaluation

The repository contains dedicated evaluation scripts for:

- Retrieval quality
- Query rewriting
- Multi-query retrieval
- Reranking
- Citation coverage
- Generation quality
- Safety classification
- Similarity analysis
- End-to-end RAG behavior
- Incremental indexing

Evaluation artifacts are preserved in the repository to document the experimentation and development process.

> Because the evidence-gating architecture evolved during development, evaluation results should be rerun after major retrieval or evidence-decision changes before historical metrics are treated as current production benchmarks.

---

## 🛡️ Safety Design

This project intentionally fails closed when critical verification steps fail.

The application does not attempt to replace healthcare professionals.

It blocks requests involving:

```text
Patient-specific diagnosis
Patient-specific treatment decisions
Patient-specific medication dosing
Medical emergencies
Attempts to override clinician advice
```

For permitted educational questions, the generated answer must still pass evidence verification.

---

## 🎯 Design Goals

```text
Evidence before generation
Verification before presentation
Citations before trust
Safety before retrieval
Fail closed when uncertain
```

The objective is not simply to generate fluent medical answers.

The objective is to produce answers that can be traced back to the available evidence.

---

## 🔮 Future Improvements

- Structured backend API
- Persistent optional conversation sessions
- Additional medical sources
- Cross-document retrieval
- Improved reranking models
- Automated evaluation in CI/CD
- Retrieval observability dashboard
- Admin knowledge-base management interface
- Production monitoring and logging

---

## ⚠️ Medical Disclaimer

This project is intended for research, educational, and demonstration purposes.

It is not a medical device and does not provide medical diagnosis, personalized treatment recommendations, medication prescriptions, or emergency medical guidance.

For individual medical concerns, users should consult an appropriate qualified healthcare professional.

---

## 👨‍💻 Development

This repository intentionally includes evaluation scripts, intermediate processing artifacts, tests, and experimentation files.

They demonstrate the development process behind the final RAG architecture, including retrieval testing, reranking experiments, citation verification, safety evaluation, and incremental indexing.

---

## ⭐ Project Status

```text
RAG Pipeline               ✅
Supabase Vector Search     ✅
Multi-Query Retrieval      ✅
LLM Reranking              ✅
Evidence Sufficiency Judge ✅
Claim Verification         ✅
Citation Validation        ✅
Safety Classifier          ✅
Incremental Indexing       ✅
Arabic Support             ✅
Conversation Memory        ✅
Streamlit UI               ✅
Cloud Deployment           ✅
```

---

## 📜 License

This repository contains project source code and research artifacts.

Third-party documents, APIs, models, libraries, and datasets remain subject to their respective licenses and terms of use.
