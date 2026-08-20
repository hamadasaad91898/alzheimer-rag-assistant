import json
import re

from rag_chat import (
    openai_client,
    chat_model,
    rewrite_query,
    generate_multi_queries,
    multi_query_search,
    rerank_chunks,
    REFUSAL_THRESHOLD,
    FINAL_K,
)


# =========================================================
# Settings
# =========================================================

ANSWER_MAX_OUTPUT_TOKENS = 2000

JUDGE_MAX_OUTPUT_TOKENS = 600
JUDGE_MAX_ATTEMPTS = 2

MAX_CLAIMS = 6


# =========================================================
# Claim Generation Prompt
# =========================================================

CLAIM_GENERATION_PROMPT = """
You are an evidence-grounded clinical information assistant.

Answer the user's question using ONLY the retrieved
source passages provided to you.

Do not use outside knowledge.

Your answer will be automatically evaluated for
citation coverage.

IMPORTANT:

Break the answer into atomic factual claims.

Each claim must express one clear factual idea.

Every claim MUST cite one or more Chunk IDs that
directly support the WHOLE claim.

Do not cite a chunk merely because it is related.

Do not combine unrelated facts in one claim.

Do not make unsupported inferences.

Do not invent:
- Chunk IDs
- page numbers
- section names
- sources
- retrieval scores

You may only cite Chunk IDs explicitly provided
in the retrieved context.

Use at most 6 claims.

If the retrieved evidence is insufficient,
return exactly:

INSUFFICIENT_EVIDENCE


Otherwise return EXACTLY this format:

CLAIM_1: first factual claim
CITES_1: 12,13

CLAIM_2: second factual claim
CITES_2: 12

CLAIM_3: third factual claim
CITES_3: 13

Continue only as needed.

Rules:

- CLAIM numbers and CITES numbers must match.
- Every CLAIM must have a CITES line.
- Use only integer Chunk IDs.
- Do not write a bibliography.
- Do not write explanations outside this format.
""".strip()


# =========================================================
# Evidence Judge Prompt
# =========================================================

SUPPORT_JUDGE_PROMPT = """
You are a strict evidence-support verifier.

You will receive:

1. One factual claim.
2. The exact source passages cited for that claim.

Determine whether the cited passages directly support
the entire material factual content of the claim.

Rules:

- Use ONLY the provided cited passages.
- Do not use outside knowledge.
- Paraphrasing is allowed.
- The wording does not need to be identical.
- Every important factual part of the claim must be
  supported by the cited evidence.
- If only part of the claim is supported,
  return UNSUPPORTED.
- If the claim adds an inference not established by
  the evidence, return UNSUPPORTED.
- If the cited evidence is merely related but does not
  support the claim, return UNSUPPORTED.

Return exactly one word:

SUPPORTED

or

UNSUPPORTED
""".strip()


# =========================================================
# Context Builder
# =========================================================

def build_generation_context(chunks):
    parts = []

    for chunk in chunks:
        pages = chunk.get(
            "pages"
        ) or []

        pages_text = ", ".join(
            str(page)
            for page in pages
        )

        similarity = float(
            chunk.get(
                "similarity",
                0
            )
        )

        parts.append(
            f"""
CHUNK ID: {chunk["chunk_id"]}
SECTION: {chunk["section"]}
PAGES: {pages_text}
SOURCE: {chunk["source"]}
RETRIEVAL SCORE: {similarity:.4f}

CONTENT:
{chunk["content"]}
""".strip()
        )

    return "\n\n---\n\n".join(
        parts
    )


# =========================================================
# Claim Generator
# =========================================================

def generate_claims(
    question,
    chunks
):
    context = build_generation_context(
        chunks
    )

    prompt = f"""
USER QUESTION:

{question}


RETRIEVED SOURCE PASSAGES:

{context}


Answer the question using the required
CLAIM_n / CITES_n format.
""".strip()

    response = openai_client.responses.create(
        model=chat_model,
        instructions=CLAIM_GENERATION_PROMPT,
        input=prompt,
        max_output_tokens=ANSWER_MAX_OUTPUT_TOKENS
    )

    raw_output = (
        response.output_text
        or ""
    ).strip()

    if not raw_output:
        raise ValueError(
            "Claim generator returned empty output."
        )

    return raw_output


# =========================================================
# Claim Parser
# =========================================================

def parse_claims(text):
    text = (
        text or ""
    ).strip()

    if not text:
        raise ValueError(
            "Empty claim output."
        )

    if (
        text.strip().upper()
        == "INSUFFICIENT_EVIDENCE"
    ):
        return []

    claim_matches = re.findall(
        r"^CLAIM_(\d+)\s*:\s*(.+)$",
        text,
        flags=re.MULTILINE | re.IGNORECASE
    )

    cite_matches = re.findall(
        r"^CITES_(\d+)\s*:\s*(.*)$",
        text,
        flags=re.MULTILINE | re.IGNORECASE
    )

    if not claim_matches:
        raise ValueError(
            "No CLAIM_n lines found."
        )

    claims = {}

    for number, claim_text in claim_matches:
        number = int(
            number
        )

        claim_text = (
            claim_text.strip()
        )

        if claim_text:
            claims[
                number
            ] = {
                "number": number,
                "claim": claim_text,
                "cited_chunk_ids": []
            }

    citations = {}

    for number, cite_text in cite_matches:
        number = int(
            number
        )

        chunk_ids = [
            int(value)
            for value in re.findall(
                r"\d+",
                cite_text
            )
        ]

        # Remove duplicate IDs
        unique_ids = []

        for chunk_id in chunk_ids:
            if chunk_id not in unique_ids:
                unique_ids.append(
                    chunk_id
                )

        citations[
            number
        ] = unique_ids

    for number in claims:
        claims[
            number
        ][
            "cited_chunk_ids"
        ] = citations.get(
            number,
            []
        )

    ordered = [
        claims[number]
        for number in sorted(
            claims.keys()
        )
    ]

    return ordered[
        :MAX_CLAIMS
    ]


# =========================================================
# Citation Validation
# =========================================================

def validate_claim_citations(
    claims,
    retrieved_chunks
):
    valid_ids = {
        int(
            chunk["chunk_id"]
        )
        for chunk in retrieved_chunks
    }

    for claim in claims:
        cited_ids = claim[
            "cited_chunk_ids"
        ]

        valid_citations = [
            chunk_id
            for chunk_id in cited_ids
            if chunk_id in valid_ids
        ]

        invalid_citations = [
            chunk_id
            for chunk_id in cited_ids
            if chunk_id not in valid_ids
        ]

        claim[
            "valid_cited_chunk_ids"
        ] = valid_citations

        claim[
            "invalid_cited_chunk_ids"
        ] = invalid_citations

        claim[
            "has_citation"
        ] = bool(
            cited_ids
        )

        claim[
            "all_citations_valid"
        ] = (
            bool(cited_ids)
            and not invalid_citations
        )

    return claims


# =========================================================
# Cited Context Builder
# =========================================================

def build_cited_context(
    claim,
    retrieved_chunks
):
    chunk_map = {
        int(
            chunk["chunk_id"]
        ): chunk
        for chunk in retrieved_chunks
    }

    parts = []

    for chunk_id in claim[
        "valid_cited_chunk_ids"
    ]:
        chunk = chunk_map.get(
            chunk_id
        )

        if not chunk:
            continue

        parts.append(
            f"""
CHUNK ID: {chunk_id}

SECTION:
{chunk["section"]}

CONTENT:
{chunk["content"]}
""".strip()
        )

    return "\n\n---\n\n".join(
        parts
    )


# =========================================================
# Support Judge
# =========================================================

def judge_claim_support(
    claim,
    retrieved_chunks
):
    if not claim[
        "has_citation"
    ]:
        return {
            "supported": False,
            "judge_error": False,
            "raw_output": "NO_CITATION"
        }

    if not claim[
        "all_citations_valid"
    ]:
        return {
            "supported": False,
            "judge_error": False,
            "raw_output": "INVALID_CITATION"
        }

    cited_context = build_cited_context(
        claim,
        retrieved_chunks
    )

    if not cited_context:
        return {
            "supported": False,
            "judge_error": False,
            "raw_output": "NO_VALID_CONTEXT"
        }

    user_input = f"""
CLAIM:

{claim["claim"]}


CITED SOURCE PASSAGES:

{cited_context}
""".strip()

    last_error = None
    raw_output = ""

    for attempt in range(
        1,
        JUDGE_MAX_ATTEMPTS + 1
    ):
        try:
            instructions = (
                SUPPORT_JUDGE_PROMPT
            )

            if attempt > 1:
                instructions += """

IMPORTANT:

Return exactly one word.

SUPPORTED

or

UNSUPPORTED
""".strip()

            response = openai_client.responses.create(
                model=chat_model,
                instructions=instructions,
                input=user_input,
                max_output_tokens=JUDGE_MAX_OUTPUT_TOKENS
            )

            raw_output = (
                response.output_text
                or ""
            ).strip()

            normalized = (
                raw_output
                .splitlines()[0]
                .strip()
                .upper()
                if raw_output
                else ""
            )

            if normalized == "SUPPORTED":
                return {
                    "supported": True,
                    "judge_error": False,
                    "raw_output": raw_output
                }

            if normalized == "UNSUPPORTED":
                return {
                    "supported": False,
                    "judge_error": False,
                    "raw_output": raw_output
                }

            raise ValueError(
                "Judge did not return "
                "SUPPORTED or UNSUPPORTED."
            )

        except Exception as error:
            last_error = error

            print(
                f"Support judge attempt "
                f"{attempt} failed: "
                f"{error}"
            )

    return {
        "supported": False,
        "judge_error": True,
        "raw_output": raw_output,
        "error": str(
            last_error
        )
    }


# =========================================================
# Retrieval Pipeline
# =========================================================

def retrieve_for_evaluation(
    question
):
    # -----------------------------------------------------
    # Rewrite
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Multi-query
    # -----------------------------------------------------

    try:
        multi_queries = generate_multi_queries(
            question,
            rewritten_query
        )

    except Exception as error:
        print(
            "Multi-query failed:",
            error
        )

        multi_queries = [
            rewritten_query
        ]

    # -----------------------------------------------------
    # Vector retrieval
    # -----------------------------------------------------

    candidates = multi_query_search(
        multi_queries
    )

    if not candidates:
        return {
            "rewritten_query":
                rewritten_query,

            "multi_queries":
                multi_queries,

            "best_similarity":
                0.0,

            "chunks":
                [],

            "reranker_fallback":
                False
        }

    best_similarity = max(
        float(
            chunk.get(
                "similarity",
                0
            )
        )
        for chunk in candidates
    )

    # -----------------------------------------------------
    # Reranker
    # -----------------------------------------------------

    reranker_fallback = False

    try:
        reranked = rerank_chunks(
            question,
            rewritten_query,
            candidates
        )

    except Exception as error:
        print(
            "Reranker failed:",
            error
        )

        print(
            "Using vector fallback."
        )

        reranker_fallback = True

        reranked = candidates[
            :FINAL_K
        ]

    return {
        "rewritten_query":
            rewritten_query,

        "multi_queries":
            multi_queries,

        "best_similarity":
            best_similarity,

        "chunks":
            reranked,

        "reranker_fallback":
            reranker_fallback
    }


# =========================================================
# Deterministic Citation Metadata
# =========================================================

def build_citation_metadata(
    claims,
    chunks
):
    used_ids = []

    for claim in claims:
        for chunk_id in claim[
            "valid_cited_chunk_ids"
        ]:
            if chunk_id not in used_ids:
                used_ids.append(
                    chunk_id
                )

    chunk_map = {
        int(
            chunk["chunk_id"]
        ): chunk
        for chunk in chunks
    }

    citations = []

    for chunk_id in used_ids:
        chunk = chunk_map.get(
            chunk_id
        )

        if not chunk:
            continue

        citations.append(
            {
                "chunk_id":
                    chunk_id,

                "section":
                    chunk.get(
                        "section"
                    ),

                "pages":
                    chunk.get(
                        "pages"
                    ) or [],

                "source":
                    chunk.get(
                        "source"
                    ),

                "retrieval_score":
                    float(
                        chunk.get(
                            "similarity",
                            0
                        )
                    )
            }
        )

    return citations


# =========================================================
# Human-Readable Answer Preview
# =========================================================

def print_answer_preview(
    claims,
    citations
):
    print()
    print("Answer:")

    for claim in claims:
        cited = ", ".join(
            str(chunk_id)
            for chunk_id
            in claim[
                "valid_cited_chunk_ids"
            ]
        )

        if cited:
            citation_text = (
                f"[Chunk {cited}]"
            )
        else:
            citation_text = (
                "[NO VALID CITATION]"
            )

        print(
            f"- {claim['claim']} "
            f"{citation_text}"
        )

    print()
    print("Citations:")

    if not citations:
        print(
            "- None"
        )

    for citation in citations:
        print(
            f"- Chunk "
            f"{citation['chunk_id']} | "
            f"Section: "
            f"{citation['section']} | "
            f"Pages: "
            f"{citation['pages']} | "
            f"Source: "
            f"{citation['source']} | "
            f"Score: "
            f"{citation['retrieval_score']:.4f}"
        )


# =========================================================
# Main Evaluation
# =========================================================

def main():

    # -----------------------------------------------------
    # Load evaluation questions
    # -----------------------------------------------------

    with open(
        "eval_questions.json",
        "r",
        encoding="utf-8"
    ) as file:
        eval_questions = json.load(
            file
        )

    total_questions = len(
        eval_questions
    )

    detailed_results = []

    total_claims = 0

    claims_with_citation = 0
    claims_with_valid_citations = 0

    total_citation_refs = 0
    valid_citation_refs = 0
    invalid_citation_refs = 0

    supported_claims = 0
    unsupported_claims = 0

    judge_errors = 0

    refused_questions = 0
    generation_errors = 0

    fully_grounded_answers = 0
    evaluated_answers = 0

    reranker_fallbacks = 0

    # -----------------------------------------------------
    # Evaluate questions
    # -----------------------------------------------------

    for question_index, item in enumerate(
        eval_questions,
        start=1
    ):
        question = item[
            "question"
        ]

        print()
        print("=" * 70)
        print(
            f"QUESTION "
            f"{question_index}/"
            f"{total_questions}"
        )
        print("=" * 70)

        print(
            question
        )

        # =================================================
        # Retrieval
        # =================================================

        retrieval = retrieve_for_evaluation(
            question
        )

        best_similarity = retrieval[
            "best_similarity"
        ]

        chunks = retrieval[
            "chunks"
        ]

        if retrieval[
            "reranker_fallback"
        ]:
            reranker_fallbacks += 1

        print()
        print(
            f"Best Similarity: "
            f"{best_similarity:.4f}"
        )

        print(
            f"Threshold:       "
            f"{REFUSAL_THRESHOLD:.2f}"
        )

        # =================================================
        # Evidence Gate
        # =================================================

        if (
            not chunks
            or best_similarity
            < REFUSAL_THRESHOLD
        ):
            refused_questions += 1

            print(
                "Result: REFUSED"
            )

            detailed_results.append(
                {
                    "question":
                        question,

                    "status":
                        "refused",

                    "best_similarity":
                        best_similarity,

                    "claims":
                        []
                }
            )

            continue

        print(
            "Result: PASSED"
        )

        # =================================================
        # Generate Claims
        # =================================================

        try:
            raw_claim_output = generate_claims(
                question,
                chunks
            )

            claims = parse_claims(
                raw_claim_output
            )

        except Exception as error:
            generation_errors += 1

            print()
            print(
                "Claim generation error:"
            )

            print(
                error
            )

            detailed_results.append(
                {
                    "question":
                        question,

                    "status":
                        "generation_error",

                    "best_similarity":
                        best_similarity,

                    "error":
                        str(error)
                }
            )

            continue

        if not claims:
            refused_questions += 1

            print(
                "Generator returned "
                "INSUFFICIENT_EVIDENCE."
            )

            detailed_results.append(
                {
                    "question":
                        question,

                    "status":
                        "insufficient_evidence",

                    "best_similarity":
                        best_similarity,

                    "claims":
                        []
                }
            )

            continue

        evaluated_answers += 1

        # =================================================
        # Validate Citations
        # =================================================

        claims = validate_claim_citations(
            claims,
            chunks
        )

        answer_is_fully_grounded = True

        # =================================================
        # Judge Every Claim
        # =================================================

        for claim in claims:
            total_claims += 1

            cited_ids = claim[
                "cited_chunk_ids"
            ]

            valid_ids = claim[
                "valid_cited_chunk_ids"
            ]

            invalid_ids = claim[
                "invalid_cited_chunk_ids"
            ]

            if claim[
                "has_citation"
            ]:
                claims_with_citation += 1

            if claim[
                "all_citations_valid"
            ]:
                claims_with_valid_citations += 1

            total_citation_refs += len(
                cited_ids
            )

            valid_citation_refs += len(
                valid_ids
            )

            invalid_citation_refs += len(
                invalid_ids
            )

            judge_result = judge_claim_support(
                claim,
                chunks
            )

            claim[
                "supported"
            ] = judge_result[
                "supported"
            ]

            claim[
                "judge_error"
            ] = judge_result[
                "judge_error"
            ]

            claim[
                "judge_raw_output"
            ] = judge_result[
                "raw_output"
            ]

            if judge_result[
                "judge_error"
            ]:
                judge_errors += 1

            if judge_result[
                "supported"
            ]:
                supported_claims += 1

            else:
                unsupported_claims += 1
                answer_is_fully_grounded = False

            if not claim[
                "has_citation"
            ]:
                answer_is_fully_grounded = False

            if not claim[
                "all_citations_valid"
            ]:
                answer_is_fully_grounded = False

        if answer_is_fully_grounded:
            fully_grounded_answers += 1

        # =================================================
        # Build deterministic metadata
        # =================================================

        citations = build_citation_metadata(
            claims,
            chunks
        )

        # =================================================
        # Print Claim Results
        # =================================================

        print()
        print("-" * 70)
        print("CLAIM RESULTS")
        print("-" * 70)

        for claim in claims:
            print()
            print(
                f"Claim "
                f"{claim['number']}:"
            )

            print(
                claim[
                    "claim"
                ]
            )

            print(
                f"Cited IDs: "
                f"{claim['cited_chunk_ids']}"
            )

            print(
                f"Valid IDs: "
                f"{claim['valid_cited_chunk_ids']}"
            )

            print(
                f"Invalid IDs: "
                f"{claim['invalid_cited_chunk_ids']}"
            )

            print(
                f"Supported: "
                f"{claim['supported']}"
            )

        print_answer_preview(
            claims,
            citations
        )

        print()
        print(
            "Fully Grounded Answer: "
            f"{answer_is_fully_grounded}"
        )

        # =================================================
        # Save Detail
        # =================================================

        detailed_results.append(
            {
                "question":
                    question,

                "status":
                    "evaluated",

                "best_similarity":
                    best_similarity,

                "rewritten_query":
                    retrieval[
                        "rewritten_query"
                    ],

                "multi_queries":
                    retrieval[
                        "multi_queries"
                    ],

                "retrieved_chunk_ids": [
                    int(
                        chunk["chunk_id"]
                    )
                    for chunk in chunks
                ],

                "raw_claim_output":
                    raw_claim_output,

                "claims":
                    claims,

                "citations":
                    citations,

                "fully_grounded":
                    answer_is_fully_grounded
            }
        )

    # =====================================================
    # Final Metrics
    # =====================================================

    citation_presence_coverage = (
        claims_with_citation
        / total_claims
        if total_claims
        else 0.0
    )

    claim_valid_citation_rate = (
        claims_with_valid_citations
        / total_claims
        if total_claims
        else 0.0
    )

    citation_reference_validity = (
        valid_citation_refs
        / total_citation_refs
        if total_citation_refs
        else 0.0
    )

    claim_support_rate = (
        supported_claims
        / total_claims
        if total_claims
        else 0.0
    )

    fully_grounded_answer_rate = (
        fully_grounded_answers
        / evaluated_answers
        if evaluated_answers
        else 0.0
    )

    end_to_end_grounded_rate = (
        fully_grounded_answers
        / total_questions
        if total_questions
        else 0.0
    )

    answer_generation_rate = (
        evaluated_answers
        / total_questions
        if total_questions
        else 0.0
    )

    # =====================================================
    # Print Final Summary
    # =====================================================

    print()
    print("=" * 70)
    print("CITATION COVERAGE EVALUATION")
    print("=" * 70)

    print(
        f"Total Questions:             "
        f"{total_questions}"
    )

    print(
        f"Evaluated Answers:           "
        f"{evaluated_answers}"
    )

    print(
        f"Refused / Insufficient:      "
        f"{refused_questions}"
    )

    print(
        f"Generation Errors:           "
        f"{generation_errors}"
    )

    print(
        f"Reranker Fallbacks:          "
        f"{reranker_fallbacks}"
    )

    print()

    print(
        f"Total Claims:                "
        f"{total_claims}"
    )

    print(
        f"Claims With Citation:        "
        f"{claims_with_citation}"
    )

    print(
        f"Claims With Valid Citations: "
        f"{claims_with_valid_citations}"
    )

    print(
        f"Supported Claims:            "
        f"{supported_claims}"
    )

    print(
        f"Unsupported Claims:          "
        f"{unsupported_claims}"
    )

    print(
        f"Judge Errors:                "
        f"{judge_errors}"
    )

    print()

    print(
        f"Citation Presence Coverage:  "
        f"{citation_presence_coverage:.4f}"
    )

    print(
        f"Claim Valid Citation Rate:   "
        f"{claim_valid_citation_rate:.4f}"
    )

    print(
        f"Citation Reference Validity: "
        f"{citation_reference_validity:.4f}"
    )

    print(
        f"Claim Support Rate:          "
        f"{claim_support_rate:.4f}"
    )

    print()

    print(
        f"Fully Grounded Answers:      "
        f"{fully_grounded_answers}"
        f"/{evaluated_answers}"
    )

    print(
        f"Fully Grounded Answer Rate:  "
        f"{fully_grounded_answer_rate:.4f}"
    )

    print(
        f"Answer Generation Rate:      "
        f"{answer_generation_rate:.4f}"
    )

    print(
        f"End-to-End Grounded Rate:    "
        f"{end_to_end_grounded_rate:.4f}"
    )

    # =====================================================
    # Save Results
    # =====================================================

    output = {
        "settings": {
            "refusal_threshold":
                REFUSAL_THRESHOLD,

            "final_k":
                FINAL_K,

            "max_claims":
                MAX_CLAIMS,

            "generator_model":
                chat_model,

            "support_judge_model":
                chat_model,

            "citation_metadata_source":
                "python_from_retrieved_chunks"
        },

        "summary": {
            "total_questions":
                total_questions,

            "evaluated_answers":
                evaluated_answers,

            "refused_or_insufficient":
                refused_questions,

            "generation_errors":
                generation_errors,

            "reranker_fallbacks":
                reranker_fallbacks,

            "total_claims":
                total_claims,

            "claims_with_citation":
                claims_with_citation,

            "claims_with_valid_citations":
                claims_with_valid_citations,

            "supported_claims":
                supported_claims,

            "unsupported_claims":
                unsupported_claims,

            "judge_errors":
                judge_errors,

            "citation_presence_coverage":
                citation_presence_coverage,

            "claim_valid_citation_rate":
                claim_valid_citation_rate,

            "citation_reference_validity":
                citation_reference_validity,

            "claim_support_rate":
                claim_support_rate,

            "fully_grounded_answers":
                fully_grounded_answers,

            "fully_grounded_answer_rate":
                fully_grounded_answer_rate,

            "answer_generation_rate":
                answer_generation_rate,

            "end_to_end_grounded_rate":
                end_to_end_grounded_rate
        },

        "questions":
            detailed_results
    }

    with open(
        "citation_coverage_evaluation.json",
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
        "citation_coverage_evaluation.json"
    )

    print("=" * 70)


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    main()