
import pytest

from ..semantic_search import (
    SemanticSearchEngine,
)


@pytest.fixture
def engine():
    search_engine = SemanticSearchEngine()

    search_engine.add_document(
        "Atlas handles transient provider failures using retries.",
        [0.9, 0.8, 0.1],
    )
    search_engine.add_document(
        "Atlas uses pytest for automated testing.",
        [0.1, 0.2, 0.9],
    )
    search_engine.add_document(
        "Exponential backoff helps recover from API failures.",
        [0.8, 0.9, 0.2],
    )

    return search_engine


@pytest.fixture
def query_embedding():
    return [0.85, 0.85, 0.15]


def test_add_document(engine):
    assert len(engine.documents) == 3

    document = engine.documents[0]

    assert document["id"] == 1
    assert document["text"] == (
        "Atlas handles transient provider failures using retries."
    )
    assert document["embedding"] == [0.9, 0.8, 0.1]


def test_document_ids_are_incremental():
    search_engine = SemanticSearchEngine()

    search_engine.add_document("First document", [1, 0])
    search_engine.add_document("Second document", [0, 1])

    assert [doc["id"] for doc in search_engine.documents] == [1, 2]


def test_search_returns_results_in_descending_similarity_order(
    engine, query_embedding
):
    results = engine.search(query_embedding)

    scores = [result["score"] for result in results]

    assert scores == sorted(scores, reverse=True)


def test_search_returns_most_similar_documents(
    engine, query_embedding
):
    results = engine.search(query_embedding, top_k=2)

    assert len(results) == 2
    assert [result["id"] for result in results] == [3, 1]


def test_search_returns_top_k_results(engine, query_embedding):
    results = engine.search(query_embedding, top_k=2)

    assert len(results) == 2


def test_search_returns_all_results_when_top_k_exceeds_document_count(
    engine, query_embedding
):
    results = engine.search(query_embedding, top_k=10)

    assert len(results) == 3


def test_search_with_zero_top_k_returns_empty_list(
    engine, query_embedding
):
    assert engine.search(query_embedding, top_k=0) == []


def test_search_with_negative_top_k_raises_value_error(
    engine, query_embedding
):
    with pytest.raises(ValueError):
        engine.search(query_embedding, top_k=-1)


@pytest.mark.parametrize("top_k", ["2", 2.5, True])
def test_search_with_invalid_top_k_raises_type_error(
    engine, query_embedding, top_k
):
    with pytest.raises(TypeError):
        engine.search(query_embedding, top_k=top_k)


def test_search_with_empty_document_collection(query_embedding):
    search_engine = SemanticSearchEngine()

    assert search_engine.search(query_embedding) == []


def test_search_does_not_mutate_document_collection(
    engine, query_embedding
):
    original_documents = [document.copy() for document in engine.documents]

    engine.search(query_embedding, top_k=2)

    assert engine.documents == original_documents


def test_search_engines_have_independent_document_collections():
    engine_one = SemanticSearchEngine()
    engine_two = SemanticSearchEngine()

    engine_one.add_document("First document", [1, 0])

    assert len(engine_one.documents) == 1
    assert engine_two.documents == []


def test_search_with_mismatched_dimensions_raises_value_error(engine):
    with pytest.raises(ValueError):
        engine.search([0.1, 0.2])


def test_search_with_zero_query_vector_raises_value_error(engine):
    with pytest.raises(ValueError):
        engine.search([0, 0, 0])
