# Final End-to-End Evaluation Report

## Final Status

**PASS**

## Production Pipeline Evaluated

Question → Safety Classifier → Query Rewrite → Multi-Query Retrieval → Evidence Gate → Reranker → Atomic Claim Generation → Citation Validation → Evidence Judge → Unsupported Claim Removal → Citation Metadata → Confidence → Final Answer / Refusal

## Fresh End-to-End Metrics

- Total cases: 64
- Overall routing accuracy: 100.00%
- In-scope answer success: 100.00%
- Out-of-scope refusal rate: 100.00%
- Hard-negative refusal rate: 100.00%
- Safety block rate: 100.00%
- Safety category accuracy: 100.00%
- Arabic routing accuracy: 100.00%
- Claim verification rate: 100.00%
- Generated claims: 140
- Verified claims: 140
- Rejected claims: 0
- System errors: 0
- Generation errors: 0
- Reranker fallbacks: 0
- Evidence judge errors: 0
- Unsafe requests allowed: 0

## Fresh Retrieval Metrics

- Questions evaluated: 20
- Hit Rate@5: 1.0000
- Mean Precision@5: 0.4900
- Mean Recall@5: 0.9875
- MRR: 1.0000
- Mean nDCG@5: 0.9854

## Previous Component Evaluations

### Retrieval

- hit_rate_at_5: 1.0000
- mean_precision_at_5: 0.4500
- mean_recall_at_5: 0.9292
- mrr: 0.9750
- mean_ndcg_at_5: 0.9185

### Threshold Calibration

- lowest_in_scope_score: 0.5643
- highest_out_of_scope_score: 0.5208
- average_in_scope_score: 0.6441
- average_out_of_scope_score: 0.2574
- Production threshold: 0.54

### Citation / Grounding

- citation_reference_validity: 1.0000
- claim_support_rate: 0.9908
- fully_grounded_answer_rate: 0.9500

### Generation Quality

- average_overall: 4.9600
- average_faithfulness: 5.0000
- strict_pass_rate: 1.0000
- core_pass_rate: N/A

## Quality Gates

| Gate | Value | Target | Result |
|---|---|---|---|
| Fresh Retrieval Hit Rate@5 | 1.0000 | >= 0.95 | PASS |
| Fresh Retrieval Mean Recall@5 | 0.9875 | >= 0.90 | PASS |
| Fresh Retrieval Mean nDCG@5 | 0.9854 | >= 0.90 | PASS |
| Evidence Threshold Separation | `{"highest_out_of_scope": 0.520799987069411, "threshold": 0.54, "lowest_in_scope": 0.564292127471349}` | highest_out_of_scope < threshold <= lowest_in_scope | PASS |
| Citation Reference Validity | 1.0000 | >= 0.99 | PASS |
| Offline Claim Support Rate | 0.9908 | >= 0.95 | PASS |
| Fully Grounded Answer Rate | 0.9500 | >= 0.95 | PASS |
| Generation Overall Quality | 4.9600 | >= 4.50 / 5 | PASS |
| Generation Faithfulness | 5.0000 | >= 4.50 / 5 | PASS |
| Generation Strict Pass Rate | 1.0000 | >= 0.95 | PASS |
| Overall E2E Routing Accuracy | 1.0000 | >= 0.95 | PASS |
| In-Scope Answer Success | 1.0000 | >= 0.95 | PASS |
| Out-of-Scope Refusal Rate | 1.0000 | >= 0.95 | PASS |
| Hard-Negative Refusal Rate | 1.0000 | >= 0.75 | PASS |
| Safety Block Rate | 1.0000 | = 1.00 | PASS |
| Safety Category Accuracy | 1.0000 | >= 0.95 | PASS |
| Arabic Routing Accuracy | 1.0000 | >= 0.95 | PASS |
| Fresh Claim Verification Rate | 1.0000 | >= 0.95 | PASS |
| System Errors | 0 | = 0 | PASS |
| Unsafe Requests Allowed | 0 | = 0 | PASS |
| Evidence Judge Errors | 0 | = 0 | PASS |

## Failed / Unexpected Cases

None.
## Evaluation Note

Generation and evidence-verification components use LLM-based automated evaluation. These results are appropriate for internal engineering validation but do not replace independent human clinical validation.
