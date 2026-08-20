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
