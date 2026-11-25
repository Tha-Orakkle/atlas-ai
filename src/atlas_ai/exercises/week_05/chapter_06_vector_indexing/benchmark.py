import time

import faiss
import numpy as np


def generate_dataset(
    num_vectors=10_000,
    num_queries=100,
    dimension=128,
    seed=42
):
    rng = np.random.default_rng(seed)

    vectors = rng.standard_normal(
        size=(num_vectors, dimension),
        dtype=np.float32
    )

    queries = rng.standard_normal(
        size=(num_queries, dimension),
        dtype=np.float32
    )

    vectors = np.ascontiguousarray(vectors)
    queries = np.ascontiguousarray(queries)

    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    return vectors, queries


def build_exact_index(vectors):
    dimension = vectors.shape[-1]

    index = faiss.IndexFlatIP(dimension)

    start_time = time.perf_counter()

    index.add(vectors)

    build_time = time.perf_counter() - start_time

    return index, build_time


def build_hnsw_index(vectors, m=32, ef_construction=100, ef_search=64):
    dimension = vectors.shape[-1]

    index = faiss.IndexHNSWFlat(
        dimension,
        m,
        faiss.METRIC_INNER_PRODUCT
    )

    # Configure graph construction before adding vectors.
    index.hnsw.efConstruction = ef_construction

    start_time = time.perf_counter()

    index.add(vectors)

    build_time = time.perf_counter() - start_time

    # Configure the search effort.
    index.hnsw.efSearch = ef_search

    return index, build_time


def search(index, queries, top_k=10):
    start_time = time.perf_counter()

    distances, indices = index.search(queries, top_k)

    elapsed_time = time.perf_counter() - start_time

    average_latency_ms = (
        elapsed_time / len(queries)
    ) * 1000

    return {
        "distances": distances,
        "indices": indices,
        "total_time": elapsed_time,
        "average_latency_ms": average_latency_ms
    }


def calculate_recall_at_k(exact_indices, approximate_indices, k=10):
    if exact_indices.shape != approximate_indices.shape:
        raise ValueError("Result arrays must have identical shapes")

    if exact_indices.ndim != 2:
        raise ValueError("Result arrays must be two-dimensional")

    if k < 0:
        raise ValueError("k must be greater than zero")

    if k > exact_indices.shape[-1]:
        raise ValueError("k cannot exceed the number of returned results")

    recalls = []

    for exact_row, approximate_row in zip(
        exact_indices,
        approximate_indices
    ):
        exact_set = set(exact_row[:k])
        approximate_set = set(approximate_row[:k])

        overlap = len(exact_set.intersection(approximate_set))

        recalls.append(overlap / k)

    return float(np.mean(recalls))


if __name__ == "__main__":
    vectors, queries = generate_dataset()

    # Stage B: Exact search
    exact_index, exact_build_time = build_exact_index(vectors)

    exact_results = search(
        exact_index,
        queries,
        top_k=10,
    )

    # Stage C: Approximate search
    hnsw_index, hnsw_build_time = build_hnsw_index(
        vectors,
        m=32,
        ef_construction=100,
        ef_search=256,
    )

    hnsw_results = search(
        hnsw_index,
        queries,
        top_k=10,
    )

    # Stage D: Retrieval quality
    recall = calculate_recall_at_k(
        exact_results["indices"],
        hnsw_results["indices"],
        k=10,
    )

    # Stage E: Comparison
    print("\n===== EXACT SEARCH =====")
    print(f"Index size: {exact_index.ntotal}")
    print(f"Build time: {exact_build_time * 1000:.3f} ms")
    print(
        f"Average query latency: "
        f"{exact_results['average_latency_ms']:.3f} ms"
    )

    print("\n===== HNSW SEARCH =====")
    print(f"Index size: {hnsw_index.ntotal}")
    print(f"Build time: {hnsw_build_time * 1000:.3f} ms")
    print(
        f"Average query latency: "
        f"{hnsw_results['average_latency_ms']:.3f} ms"
    )

    print("\n===== RETRIEVAL QUALITY =====")
    print(f"Recall@10: {recall:.4f}")
    print(f"Recall@10 percentage: {recall * 100:.2f}%")
