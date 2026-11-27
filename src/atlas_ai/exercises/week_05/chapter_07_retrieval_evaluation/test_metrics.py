from .metrics import precision_at_k, recall_at_k


def test_precision_ignores_relevant_documents_outside_top_k():
    retrieved = ["doc_1", "doc_2", "doc_3"]
    relevant = {"doc_3"}

    assert precision_at_k(retrieved, relevant, 2) == 0.0


def test_precision_with_generator():
    retrieved = (doc for doc in ["doc_1", "doc_2"])
    relevant = {"doc_1"}

    assert precision_at_k(retrieved, relevant, 2) == 0.5


def test_recall_with_generator():
    retrieved = (doc for doc in ["doc_1", "doc_2"])
    relevant = {"doc_1", "doc_3"}

    assert recall_at_k(retrieved, relevant, 2) == 0.5
