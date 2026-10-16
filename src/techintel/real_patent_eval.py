"""Curated real-patent *summary* retrieval smoke evaluation.

The document identifiers are real published patent IDs but records are manual
paraphrases, NOT patent full texts. Question labels were written by the same
project author and must not be treated as independent external validation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .patents import search_curated_patent_notes
from .retrieval import KnowledgeIndex

BASE = Path(__file__).resolve().parents[2]


def evaluate_real_patent_notes() -> dict:
    index = KnowledgeIndex([BASE / "corpus" / "patents_source_notes"])
    cases = [json.loads(line) for line in (BASE / "evals" / "real_patent_notes_gold.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    positives = [x for x in cases if x["answerable"]]
    negatives = [x for x in cases if not x["answerable"]]
    report = {"corpus": "3 real patent IDs, author-paraphrased summaries, NOT full patent claims",
              "labeling": "13 author-written questions, no independent test-set annotation",
              "n_documents": index.document_count, "n_queries": len(cases), "positive_queries": len(positives),
              "negative_queries": len(negatives), "methods": {}}
    for method in ("word", "hybrid", "bm25"):
        for min_terms in (0, 2):
            ranks = []
            correct_negative = 0
            failures = []
            for case in cases:
                result = search_curated_patent_notes(index, case["query"], k=3,
                                                     retrieval=method, min_content_terms=min_terms)
                ids = [x["doc_id"] for x in result["results"]]
                if case["answerable"]:
                    rank = next((i + 1 for i, doc_id in enumerate(ids) if doc_id in case["gold_docs"]), None)
                    ranks.append(rank)
                    if rank is None:
                        failures.append({"query": case["query"], "issue": "missed_positive", "returned": ids})
                else:
                    if not ids:
                        correct_negative += 1
                    else:
                        failures.append({"query": case["query"], "issue": "retrieved_on_unknown", "returned": ids})
            metrics = {
                "hit_at_1": round(sum(x == 1 for x in ranks) / len(positives), 4),
                "hit_at_3": round(sum(x is not None for x in ranks) / len(positives), 4),
                "mrr": round(sum(1/x for x in ranks if x) / len(positives), 4),
                "ood_correct_rejection": round(correct_negative / len(negatives), 4),
                "failures": failures,
            }
            if min_terms == 0:
                report.setdefault("ablation_without_evidence_gate", {})[method] = metrics
            else:
                report["methods"][method] = metrics
    report["limitations"] = [
        "Three short summaries authored by the evaluator, rather than actual patent full texts.",
        "All questions and gold doc IDs were written with knowledge of the corpus: data leakage risk.",
        "OOD rule is empty retrieval, not calibrated legal evidence sufficiency.",
        "This is not Ragas Faithfulness, real-model answer accuracy or patent novelty evaluation.",
    ]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()
    result = evaluate_real_patent_notes()
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
