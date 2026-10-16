import numpy as np
import pytest
from pathlib import Path
from techintel.retrieval import KnowledgeIndex
from techintel.retrieval_enhanced import DenseSparseRRF


def toy_index(tmp_path):
    (tmp_path / "a.md").write_text("# Gripper\nrobot gripper tactile sensor array", encoding="utf8")
    (tmp_path / "b.md").write_text("# Battery\nsulfide solid electrolyte moisture humidity", encoding="utf8")
    return KnowledgeIndex([tmp_path])


def fake_encoder(texts):
    return np.asarray([[1., 0.] if ("gripper" in x.lower() or "robot" in x.lower())
                       else [0., 1.] for x in texts], dtype=np.float32)


def test_rrf_with_true_vector_backend_mocked(tmp_path):
    index = toy_index(tmp_path)
    pipe = DenseSparseRRF(index, embed=fake_encoder)
    results = pipe.search("robot gripper", k=2, min_dense_score=0.0)
    assert results[0].doc_id == "a"
    assert set(h.doc_id for h in results) == {"a", "b"}


def test_reranker_changes_order_and_input_validation(tmp_path):
    index = toy_index(tmp_path)
    pipe = DenseSparseRRF(index, embed=fake_encoder,
                          rerank=lambda query, docs: [float(i) for i, _ in enumerate(docs)])
    results = pipe.search("robot gripper", k=2, min_dense_score=0.0)
    assert results[0].doc_id == "b"
    with pytest.raises(ValueError):
        pipe.search("robot", k=4, pool_size=2)
    bad = DenseSparseRRF(index, embed=fake_encoder, rerank=lambda q, docs: [1.0])
    with pytest.raises(ValueError, match="one numeric"):
        bad.search("robot gripper", k=2, min_dense_score=0.0)
