"""Tiny synthetic patent retrieval benchmark, independent from production claims."""
from __future__ import annotations
import json
from pathlib import Path

from .retrieval import KnowledgeIndex
from .patents import search_local_patents

BASE = Path(__file__).resolve().parents[2]


def evaluate_patent_demo() -> dict:
    index = KnowledgeIndex([BASE / 'corpus' / 'patents_demo'])
    cases = [json.loads(s) for s in (BASE / 'evals' / 'patent_gold.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    report = {'dataset': '16 manually written queries on FOUR fictional patent teaching records',
              'n_documents': index.document_count,
              'n_queries': len(cases),
              'methods': {}}
    for method in ('word', 'hybrid', 'bm25'):
        matched_at_1 = matched_at_3 = reciprocal = abstain = 0
        fails = []
        answerable = sum(x['answerable'] for x in cases)
        ood = len(cases) - answerable
        for case in cases:
            result = search_local_patents(index, case['query'], k=3, retrieval=method)
            ids = [x['doc_id'] for x in result['results']]
            if case['answerable']:
                ranks = [i+1 for i, x in enumerate(ids) if x in case['gold_docs']]
                if ranks:
                    matched_at_3 += 1
                    reciprocal += 1/min(ranks)
                    if min(ranks) == 1:
                        matched_at_1 += 1
                else:
                    fails.append({'query':case['query'], 'returned':ids, 'gold':case['gold_docs']})
            else:
                if not ids:
                    abstain += 1
                else:
                    fails.append({'query':case['query'], 'returned':ids, 'gold':[]})
        report['methods'][method] = {'hit_at_1':round(matched_at_1/answerable,4),
                                     'hit_at_3':round(matched_at_3/answerable,4),
                                     'mrr':round(reciprocal/answerable,4),
                                     'ood_correct_rejection':round(abstain/ood,4),
                                     'failures':fails}
    report['limitations'] = ['Synthetic, tiny dataset. Author-written queries and targets can leak keywords.',
                             'Claim parser and retrieval metrics are NOT patent novelty evaluation.',
                             'Real patents, human annotation and independent held-out testing are necessary.',
                             'No dense model, cross-encoder reranker, LoRA or live patent endpoint measured.']
    return report


def main():
    print(json.dumps(evaluate_patent_demo(), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
