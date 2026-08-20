import json
import re
from statistics import mean

from rag_chat import (
    openai_client,
    chat_model,
    classify_safety,
    rewrite_query,
    generate_multi_queries,
    multi_query_search,
    rerank_chunks,
    generate_claims,
    parse_claims,
    verify_claims,
    build_citation_metadata,
    render_verified_answer,
    REFUSAL_THRESHOLD,
    FINAL_K,
)


# =========================================================
# Settings
# =========================================================

JUDGE_MAX_ATTEMPTS = 2
JUDGE_MAX_OUTPUT_TOKENS = 1200


# =========================================================
# Generation Quality Judge
# =========================================================

GENERATION_JUDGE_PROMPT = """
You are a strict evaluator of answer-generation quality
for an evidence-grounded Alzheimer's disease RAG system.

You will receive:

1. The user's question.
2. The retrieved evidence passages available to the generator.
3. The final verified answer shown to the user.

Evaluate ONLY the quality of the generated answer.

Do NOT evaluate retrieval quality.
Do NOT penalize the answer for information that is not
present in the supplied evidence.

Use ONLY the supplied evidence when assessing factual
support and correctness.

Score each dimension from 0 to 5 using integers only.


=========================================================
1. RELEVANCE
=========================================================

5:
Directly answers the exact question with no meaningful drift.

4:
Answers the question well with minor unnecessary information.

3:
Partially answers the question or contains noticeable drift.

2:
Only weakly addresses the requested information.

1:
Mostly unrelated.

0:
Does not answer the question.


=========================================================
2. COMPLETENESS
=========================================================

Judge completeness relative ONLY to the important
answer-worthy information available in the supplied evidence.

5:
Covers essentially all major evidence-supported points needed
for a strong answer.

4:
Covers most important points but misses a minor useful point.

3:
Covers the central idea but misses several important points.

2:
Substantially incomplete.

1:
Only one small part of the available answer is provided.

0:
No meaningful answer.


=========================================================
3. FAITHFULNESS
=========================================================

5:
Every substantive factual statement is directly supported
by the supplied evidence.

4:
Almost completely grounded, with only a very minor
unsupported implication.

3:
Mostly grounded but contains one noticeable unsupported
statement or inference.

2:
Several claims are not adequately supported.

1:
Much of the answer relies on unsupported information.

0:
The answer is largely hallucinated or contradicted by
the evidence.


=========================================================
4. CORRECTNESS
=========================================================

Assess whether the answer accurately represents the meaning,
scope, certainty, numbers, relationships, and conclusions
of the supplied evidence.

5:
Fully accurate and precise.

4:
Accurate overall with a small imprecision that does not
materially change the meaning.

3:
Mostly correct but contains a meaningful distortion,
overstatement, understatement, or ambiguity.

2:
Contains major factual or semantic errors.

1:
Mostly incorrect.

0:
Contradicts the evidence.


=========================================================
5. CLARITY
=========================================================

This includes clarity, directness, organization,
and appropriate conciseness.

5:
Very clear, direct, well organized, and appropriately concise.

4:
Clear overall with minor redundancy or awkward wording.

3:
Understandable but noticeably verbose, repetitive,
poorly organized, or vague.

2:
Difficult to follow.

1:
Very confusing.

0:
Unusable.


=========================================================
IMPORTANT RULES
=========================================================

- Do not use outside medical knowledge.
- Do not reward the answer simply for having citations.
- Judge whether the actual answer content is good.
- Do not assume a claim is correct unless the supplied
  evidence supports it.
- A concise answer can still receive 5 for completeness
  if it covers the major evidence-supported points.
- Do not penalize formatting metadata such as confidence,
  citation IDs, or retrieval scores.
- Evaluate the answer content itself.


Return EXACTLY this format:

RELEVANCE: 5
COMPLETENESS: 5
FAITHFULNESS: 5
CORRECTNESS: 5
CLARITY: 5
CRITICAL_ISSUE: NONE
RATIONALE: concise explanation

No JSON.
No markdown.
No additional headings.
""".strip()


# =========================================================
# Evidence Context
# =========================================================

def build_judge_evidence(chunks):
    parts = []

    for chunk in chunks:
        pages = (
            chunk.get("pages")
            or []
        )

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
RETRIEVAL SCORE: {similarity:.4f}

CONTENT:
{chunk["content"]}
""".strip()
        )

    return "\n\n---\n\n".join(
        parts
    )


# =========================================================
# Extract Answer Body
# =========================================================

def extract_answer_body(final_answer):
    """
    Extract only the user-visible Answer section.

    Ignore:
    - Supporting Evidence
    - Citation metadata
    - Confidence
    - Safety note

    because those are evaluated separately.
    """

    text = (
        final_answer
        or ""
    ).strip()

    match = re.search(
        r"Answer:\s*(.*?)"
        r"(?:\n\s*Supporting Evidence:|\Z)",
        text,
        flags=(
            re.IGNORECASE
            | re.DOTALL
        )
    )

    if match:
        return (
            match
            .group(1)
            .strip()
        )

    return text


# =========================================================
# Judge Parser
# =========================================================

def parse_score(
    text,
    name
):
    match = re.search(
        rf"^{name}\s*:\s*([0-5])\s*$",
        text,
        flags=(
            re.MULTILINE
            | re.IGNORECASE
        )
    )

    if not match:
        raise ValueError(
            f"Missing or invalid "
            f"{name} score."
        )

    return int(
        match.group(1)
    )


def parse_generation_judgment(text):
    text = (
        text
        or ""
    ).strip()

    if not text:
        raise ValueError(
            "Judge returned empty output."
        )

    relevance = parse_score(
        text,
        "RELEVANCE"
    )

    completeness = parse_score(
        text,
        "COMPLETENESS"
    )

    faithfulness = parse_score(
        text,
        "FAITHFULNESS"
    )

    correctness = parse_score(
        text,
        "CORRECTNESS"
    )

    clarity = parse_score(
        text,
        "CLARITY"
    )

    issue_match = re.search(
        r"^CRITICAL_ISSUE\s*:\s*(.*)$",
        text,
        flags=(
            re.MULTILINE
            | re.IGNORECASE
        )
    )

    rationale_match = re.search(
        r"^RATIONALE\s*:\s*(.*)$",
        text,
        flags=(
            re.MULTILINE
            | re.IGNORECASE
        )
    )

    critical_issue = (
        issue_match
        .group(1)
        .strip()
        if issue_match
        else "UNKNOWN"
    )

    rationale = (
        rationale_match
        .group(1)
        .strip()
        if rationale_match
        else ""
    )

    scores = [
        relevance,
        completeness,
        faithfulness,
        correctness,
        clarity
    ]

    overall = (
        sum(scores)
        / len(scores)
    )

    # Strict quality pass:
    # every metric must be at least 4.
    strict_pass = all(
        score >= 4
        for score in scores
    )

    # Core pass:
    # overall quality >= 4,
    # and the three most important dimensions
    # must each be >= 4.
    core_pass = (
        overall >= 4.0
        and relevance >= 4
        and faithfulness >= 4
        and correctness >= 4
    )

    return {
        "relevance":
            relevance,

        "completeness":
            completeness,

        "faithfulness":
            faithfulness,

        "correctness":
            correctness,

        "clarity":
            clarity,

        "overall":
            overall,

        "strict_pass":
            strict_pass,

        "core_pass":
            core_pass,

        "critical_issue":
            critical_issue,

        "rationale":
            rationale
    }


# =========================================================
# Generation Judge
# =========================================================

def judge_generation(
    question,
    chunks,
    answer_body
):
    evidence = build_judge_evidence(
        chunks
    )

    judge_input = f"""
USER QUESTION:

{question}


AVAILABLE RETRIEVED EVIDENCE:

{evidence}


FINAL GENERATED ANSWER:

{answer_body}
""".strip()

    last_error = None

    for attempt in range(
        1,
        JUDGE_MAX_ATTEMPTS + 1
    ):
        raw_output = ""

        try:
            instructions = (
                GENERATION_JUDGE_PROMPT
            )

            if attempt > 1:
                instructions += """

IMPORTANT RETRY:

Return exactly these seven lines:

RELEVANCE: <0-5>
COMPLETENESS: <0-5>
FAITHFULNESS: <0-5>
CORRECTNESS: <0-5>
CLARITY: <0-5>
CRITICAL_ISSUE: <NONE or short issue>
RATIONALE: <one concise sentence>
""".strip()

            response = (
                openai_client.responses.create(
                    model=chat_model,
                    instructions=instructions,
                    input=judge_input,
                    max_output_tokens=(
                        JUDGE_MAX_OUTPUT_TOKENS
                    )
                )
            )

            raw_output = (
                response.output_text
                or ""
            ).strip()

            result = (
                parse_generation_judgment(
                    raw_output
                )
            )

            result[
                "raw_output"
            ] = raw_output

            result[
                "judge_error"
            ] = False

            return result

        except Exception as error:
            last_error = error

            print(
                f"Generation judge attempt "
                f"{attempt} failed: "
                f"{error}"
            )

            print(
                "Raw output:",
                repr(raw_output)
            )

    return {
        "judge_error":
            True,

        "error":
            str(last_error),

        "raw_output":
            ""
    }


# =========================================================
# Production Pipeline For Evaluation
# =========================================================

def run_production_pipeline(
    question
):
    # -----------------------------------------------------
    # 1. Safety
    # -----------------------------------------------------

    safety = classify_safety(
        question
    )

    if not safety[
        "allowed"
    ]:
        return {
            "status":
                "safety_refused",

            "safety":
                safety
        }

    # -----------------------------------------------------
    # 2. Query Rewrite
    # -----------------------------------------------------

    try:
        rewritten_query = rewrite_query(
            question
        )

    except Exception:
        rewritten_query = question

    # -----------------------------------------------------
    # 3. Multi-Query
    # -----------------------------------------------------

    try:
        multi_queries = (
            generate_multi_queries(
                question,
                rewritten_query
            )
        )

    except Exception:
        multi_queries = [
            rewritten_query
        ]

    # -----------------------------------------------------
    # 4. Vector Retrieval
    # -----------------------------------------------------

    candidates = multi_query_search(
        multi_queries
    )

    if not candidates:
        return {
            "status":
                "no_retrieval",

            "rewritten_query":
                rewritten_query,

            "multi_queries":
                multi_queries
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
    # 5. Evidence Gate
    # -----------------------------------------------------

    if (
        best_similarity
        < REFUSAL_THRESHOLD
    ):
        return {
            "status":
                "threshold_refused",

            "rewritten_query":
                rewritten_query,

            "multi_queries":
                multi_queries,

            "best_similarity":
                best_similarity
        }

    # -----------------------------------------------------
    # 6. Reranker
    # -----------------------------------------------------

    reranker_fallback = False

    try:
        chunks = rerank_chunks(
            question,
            rewritten_query,
            candidates
        )

    except Exception:
        reranker_fallback = True

        chunks = candidates[
            :FINAL_K
        ]

    if not chunks:
        return {
            "status":
                "no_final_chunks",

            "best_similarity":
                best_similarity
        }

    # -----------------------------------------------------
    # 7. Generate Atomic Claims
    # -----------------------------------------------------

    raw_claim_output = generate_claims(
        question,
        chunks
    )

    claims = parse_claims(
        raw_claim_output
    )

    if not claims:
        return {
            "status":
                "generation_refused",

            "best_similarity":
                best_similarity,

            "chunks":
                chunks
        }

    # -----------------------------------------------------
    # 8. Verify Claims
    # -----------------------------------------------------

    claims = verify_claims(
        claims,
        chunks
    )

    supported_claims = [
        claim
        for claim in claims
        if claim[
            "included"
        ]
    ]

    rejected_claims = [
        claim
        for claim in claims
        if not claim[
            "included"
        ]
    ]

    if not supported_claims:
        return {
            "status":
                "no_supported_claims",

            "best_similarity":
                best_similarity,

            "chunks":
                chunks,

            "claims":
                claims
        }

    # -----------------------------------------------------
    # 9. Deterministic Citations
    # -----------------------------------------------------

    citations = (
        build_citation_metadata(
            supported_claims,
            chunks
        )
    )

    # -----------------------------------------------------
    # 10. Final Production Answer
    # -----------------------------------------------------

    final_answer = (
        render_verified_answer(
            supported_claims,
            claims,
            citations,
            best_similarity
        )
    )

    answer_body = extract_answer_body(
        final_answer
    )

    return {
        "status":
            "success",

        "rewritten_query":
            rewritten_query,

        "multi_queries":
            multi_queries,

        "best_similarity":
            best_similarity,

        "reranker_fallback":
            reranker_fallback,

        "chunks":
            chunks,

        "claims":
            claims,

        "supported_claims":
            supported_claims,

        "rejected_claims":
            rejected_claims,

        "citations":
            citations,

        "final_answer":
            final_answer,

        "answer_body":
            answer_body
    }


# =========================================================
# Helpers
# =========================================================

def average_score(
    results,
    field
):
    values = [
        result[
            "judgment"
        ][
            field
        ]
        for result in results
        if (
            result.get(
                "judgment"
            )
            and not result[
                "judgment"
            ].get(
                "judge_error",
                False
            )
        )
    ]

    if not values:
        return 0.0

    return mean(
        values
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
        eval_questions = json.load(
            file
        )

    total_questions = len(
        eval_questions
    )

    results = []

    successful_answers = 0
    refused_answers = 0
    pipeline_errors = 0

    judge_errors = 0

    strict_passes = 0
    core_passes = 0

    total_generated_claims = 0
    total_supported_claims = 0
    total_rejected_claims = 0

    reranker_fallbacks = 0

    # =====================================================
    # Run Evaluation
    # =====================================================

    for index, item in enumerate(
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
            f"{index}/"
            f"{total_questions}"
        )

        print("=" * 70)

        print(
            question
        )

        try:
            pipeline = (
                run_production_pipeline(
                    question
                )
            )

        except Exception as error:
            pipeline_errors += 1

            print()
            print(
                "PIPELINE ERROR:"
            )

            print(
                error
            )

            results.append(
                {
                    "question":
                        question,

                    "status":
                        "pipeline_error",

                    "error":
                        str(error)
                }
            )

            continue

        status = pipeline[
            "status"
        ]

        if status != "success":
            refused_answers += 1

            print()
            print(
                f"Pipeline Status: "
                f"{status}"
            )

            results.append(
                {
                    "question":
                        question,

                    "status":
                        status,

                    "pipeline":
                        {
                            key: value
                            for key, value
                            in pipeline.items()
                            if key != "chunks"
                        }
                }
            )

            continue

        successful_answers += 1

        if pipeline[
            "reranker_fallback"
        ]:
            reranker_fallbacks += 1

        generated_claim_count = len(
            pipeline[
                "claims"
            ]
        )

        supported_claim_count = len(
            pipeline[
                "supported_claims"
            ]
        )

        rejected_claim_count = len(
            pipeline[
                "rejected_claims"
            ]
        )

        total_generated_claims += (
            generated_claim_count
        )

        total_supported_claims += (
            supported_claim_count
        )

        total_rejected_claims += (
            rejected_claim_count
        )

        print()
        print(
            f"Best Similarity: "
            f"{pipeline['best_similarity']:.4f}"
        )

        print(
            f"Generated Claims: "
            f"{generated_claim_count}"
        )

        print(
            f"Verified Claims:  "
            f"{supported_claim_count}"
        )

        print(
            f"Rejected Claims:  "
            f"{rejected_claim_count}"
        )

        # =================================================
        # Generation Quality Judge
        # =================================================

        judgment = judge_generation(
            question,
            pipeline[
                "chunks"
            ],
            pipeline[
                "answer_body"
            ]
        )

        if judgment.get(
            "judge_error"
        ):
            judge_errors += 1

            print()
            print(
                "GENERATION JUDGE ERROR"
            )

            print(
                judgment.get(
                    "error"
                )
            )

        else:
            if judgment[
                "strict_pass"
            ]:
                strict_passes += 1

            if judgment[
                "core_pass"
            ]:
                core_passes += 1

            print()
            print("-" * 70)
            print("GENERATION QUALITY")
            print("-" * 70)

            print(
                f"Relevance:    "
                f"{judgment['relevance']}/5"
            )

            print(
                f"Completeness: "
                f"{judgment['completeness']}/5"
            )

            print(
                f"Faithfulness: "
                f"{judgment['faithfulness']}/5"
            )

            print(
                f"Correctness:  "
                f"{judgment['correctness']}/5"
            )

            print(
                f"Clarity:      "
                f"{judgment['clarity']}/5"
            )

            print(
                f"Overall:      "
                f"{judgment['overall']:.2f}/5"
            )

            print(
                f"Strict Pass:  "
                f"{judgment['strict_pass']}"
            )

            print(
                f"Core Pass:    "
                f"{judgment['core_pass']}"
            )

            print(
                f"Critical Issue: "
                f"{judgment['critical_issue']}"
            )

            print(
                f"Rationale: "
                f"{judgment['rationale']}"
            )

        results.append(
            {
                "question":
                    question,

                "status":
                    "success",

                "best_similarity":
                    pipeline[
                        "best_similarity"
                    ],

                "rewritten_query":
                    pipeline[
                        "rewritten_query"
                    ],

                "multi_queries":
                    pipeline[
                        "multi_queries"
                    ],

                "retrieved_chunk_ids": [
                    int(
                        chunk[
                            "chunk_id"
                        ]
                    )
                    for chunk
                    in pipeline[
                        "chunks"
                    ]
                ],

                "generated_claims":
                    generated_claim_count,

                "supported_claims":
                    supported_claim_count,

                "rejected_claims":
                    rejected_claim_count,

                "answer":
                    pipeline[
                        "answer_body"
                    ],

                "judgment":
                    judgment
            }
        )

    # =====================================================
    # Valid Judgments
    # =====================================================

    judged_results = [
        result
        for result in results
        if (
            result.get(
                "status"
            ) == "success"
            and result.get(
                "judgment"
            )
            and not result[
                "judgment"
            ].get(
                "judge_error",
                False
            )
        )
    ]

    judged_count = len(
        judged_results
    )

    # =====================================================
    # Average Scores
    # =====================================================

    avg_relevance = average_score(
        judged_results,
        "relevance"
    )

    avg_completeness = average_score(
        judged_results,
        "completeness"
    )

    avg_faithfulness = average_score(
        judged_results,
        "faithfulness"
    )

    avg_correctness = average_score(
        judged_results,
        "correctness"
    )

    avg_clarity = average_score(
        judged_results,
        "clarity"
    )

    avg_overall = average_score(
        judged_results,
        "overall"
    )

    # =====================================================
    # Pass Rates
    # =====================================================

    strict_pass_rate = (
        strict_passes
        / judged_count
        if judged_count
        else 0.0
    )

    core_pass_rate = (
        core_passes
        / judged_count
        if judged_count
        else 0.0
    )

    answer_generation_rate = (
        successful_answers
        / total_questions
        if total_questions
        else 0.0
    )

    claim_verification_rate = (
        total_supported_claims
        / total_generated_claims
        if total_generated_claims
        else 0.0
    )

    # =====================================================
    # Find Worst Answer
    # =====================================================

    worst_result = None

    if judged_results:
        worst_result = min(
            judged_results,
            key=lambda result:
                result[
                    "judgment"
                ][
                    "overall"
                ]
        )

    # =====================================================
    # Final Summary
    # =====================================================

    print()
    print("=" * 70)
    print("GENERATION QUALITY EVALUATION")
    print("=" * 70)

    print(
        f"Total Questions:          "
        f"{total_questions}"
    )

    print(
        f"Successful Answers:       "
        f"{successful_answers}"
    )

    print(
        f"Refused / Insufficient:   "
        f"{refused_answers}"
    )

    print(
        f"Pipeline Errors:           "
        f"{pipeline_errors}"
    )

    print(
        f"Judge Errors:              "
        f"{judge_errors}"
    )

    print(
        f"Reranker Fallbacks:        "
        f"{reranker_fallbacks}"
    )

    print()
    print("-" * 70)
    print("AVERAGE SCORES")
    print("-" * 70)

    print(
        f"Relevance:                 "
        f"{avg_relevance:.4f}/5"
    )

    print(
        f"Completeness:              "
        f"{avg_completeness:.4f}/5"
    )

    print(
        f"Faithfulness:              "
        f"{avg_faithfulness:.4f}/5"
    )

    print(
        f"Correctness:               "
        f"{avg_correctness:.4f}/5"
    )

    print(
        f"Clarity / Conciseness:     "
        f"{avg_clarity:.4f}/5"
    )

    print(
        f"Overall Generation Score:  "
        f"{avg_overall:.4f}/5"
    )

    print()
    print("-" * 70)
    print("PASS RATES")
    print("-" * 70)

    print(
        f"Strict Pass Rate:          "
        f"{strict_pass_rate:.4f}"
    )

    print(
        f"Core Quality Pass Rate:    "
        f"{core_pass_rate:.4f}"
    )

    print(
        f"Answer Generation Rate:    "
        f"{answer_generation_rate:.4f}"
    )

    print()
    print("-" * 70)
    print("CLAIM VERIFICATION")
    print("-" * 70)

    print(
        f"Generated Claims:          "
        f"{total_generated_claims}"
    )

    print(
        f"Verified Claims:           "
        f"{total_supported_claims}"
    )

    print(
        f"Rejected Claims:           "
        f"{total_rejected_claims}"
    )

    print(
        f"Verification Rate:         "
        f"{claim_verification_rate:.4f}"
    )

    if worst_result:
        print()
        print("-" * 70)
        print("LOWEST-SCORING ANSWER")
        print("-" * 70)

        print(
            worst_result[
                "question"
            ]
        )

        print(
            f"Overall Score: "
            f"{worst_result['judgment']['overall']:.2f}/5"
        )

        print(
            f"Relevance: "
            f"{worst_result['judgment']['relevance']}/5"
        )

        print(
            f"Completeness: "
            f"{worst_result['judgment']['completeness']}/5"
        )

        print(
            f"Faithfulness: "
            f"{worst_result['judgment']['faithfulness']}/5"
        )

        print(
            f"Correctness: "
            f"{worst_result['judgment']['correctness']}/5"
        )

        print(
            f"Clarity: "
            f"{worst_result['judgment']['clarity']}/5"
        )

        print(
            "Critical Issue:"
        )

        print(
            worst_result[
                "judgment"
            ][
                "critical_issue"
            ]
        )

        print(
            "Rationale:"
        )

        print(
            worst_result[
                "judgment"
            ][
                "rationale"
            ]
        )

    # =====================================================
    # Save JSON
    # =====================================================

    output = {
        "settings": {
            "refusal_threshold":
                REFUSAL_THRESHOLD,

            "final_k":
                FINAL_K,

            "generator_model":
                chat_model,

            "judge_model":
                chat_model,

            "evaluation_type":
                "source_grounded_llm_as_judge",

            "score_scale":
                "0_to_5",

            "dimensions": [
                "relevance",
                "completeness",
                "faithfulness",
                "correctness",
                "clarity"
            ]
        },

        "summary": {
            "total_questions":
                total_questions,

            "successful_answers":
                successful_answers,

            "refused_or_insufficient":
                refused_answers,

            "pipeline_errors":
                pipeline_errors,

            "judge_errors":
                judge_errors,

            "reranker_fallbacks":
                reranker_fallbacks,

            "average_relevance":
                avg_relevance,

            "average_completeness":
                avg_completeness,

            "average_faithfulness":
                avg_faithfulness,

            "average_correctness":
                avg_correctness,

            "average_clarity":
                avg_clarity,

            "average_overall":
                avg_overall,

            "strict_pass_rate":
                strict_pass_rate,

            "core_quality_pass_rate":
                core_pass_rate,

            "answer_generation_rate":
                answer_generation_rate,

            "generated_claims":
                total_generated_claims,

            "verified_claims":
                total_supported_claims,

            "rejected_claims":
                total_rejected_claims,

            "claim_verification_rate":
                claim_verification_rate
        },

        "questions":
            results
    }

    with open(
        "generation_quality_evaluation.json",
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
        "generation_quality_evaluation.json"
    )

    print("=" * 70)


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    main()