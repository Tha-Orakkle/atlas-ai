from collections.abc import Iterable


def _validate_input(retrieved, relevant, k):
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError("k must be an integer")

    if k <= 0:
        raise ValueError("k must be a positive integer")

    if isinstance(retrieved, (str, bytes)):
        raise TypeError("retrieved must contain document IDs, not a string")

    if isinstance(relevant, (str, bytes)):
        raise TypeError("relevant must contain document IDs, not a string")

    if not isinstance(retrieved, Iterable):
        raise TypeError("retrieved must be an iterable")

    if not isinstance(relevant, Iterable):
        raise TypeError("relevant must be an iterable")

    retrieved = list(retrieved)
    relevant = list(relevant)

    try:
        if len(set(retrieved)) != len(retrieved):
            raise ValueError("retrieved contains duplicate IDs")

        if len(set(relevant)) != len(relevant):
            raise ValueError("relevant contains duplicate IDs")

    except TypeError as exc:
        raise TypeError("document IDs must be hashable") from exc

    return retrieved, relevant


def precision_at_k(retrieved, relevant, k):
    retrieved, relevant = _validate_input(retrieved, relevant, k)

    if not retrieved:
        return 0.0

    top_k = retrieved[:k]
    relevant_set = set(relevant)

    relevant_count = sum(
        document_id in relevant_set
        for document_id in top_k
    )

    return relevant_count / len(top_k)


def recall_at_k(retrieved, relevant, k):
    retrieved, relevant = _validate_input(retrieved, relevant, k)

    if not relevant:
        raise ValueError("relevant must not be empty")

    top_k = retrieved[:k]
    relevant_set = set(relevant)
    relevant_count = sum(
        document_id in relevant_set
        for document_id in top_k
    )

    return relevant_count / len(relevant_set)
