from pathlib import Path
from techintel.retrieval import KnowledgeIndex
from techintel.bm25 import BM25Index, rrf_fuse, tokenize

BASE = Path(__file__).resolve().parents[1]


def test_tokenize_bilingual():
    assert 'sulfide' in tokenize('Sulfide 电解质阻抗')
    assert '电解' in tokenize('电解质')


def test_bm25_search():
    index = KnowledgeIndex([BASE / 'corpus' / 'patents_demo'])
    result = BM25Index(index).search('robotic gripper torque tactile', k=3)
    assert result[0].doc_id == 'patent_demo_gripper'


def test_rrf_fusion_and_dedup():
    index = KnowledgeIndex([BASE / 'corpus' / 'patents_demo'])
    a = BM25Index(index).search('organic photovoltaic active layer', k=3)
    b = index.search('organic photovoltaic active layer', k=3)
    out = rrf_fuse(a, b, k=3)
    assert len({h.id for h in out}) == len(out)
    assert out[0].doc_id == 'patent_demo_organic_pv'
