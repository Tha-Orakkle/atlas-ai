import pytest

from atlas_ai.vector_math import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
    magnitude,
)


def test_dot_product():
    assert dot_product([1, 2], [3, 4]) == 11


def test_magnitude():
    assert magnitude([3, 4]) == 5


def test_cosine_similarity_identical_direction():
    assert cosine_similarity([1, 2], [2, 4]) == pytest.approx(1.0)


def test_euclidean_distance():
    assert euclidean_distance([1, 2], [4, 6]) == 5


def test_mismatched_dimensions():
    with pytest.raises(ValueError):
        dot_product([1, 2], [3])


def test_zero_vector_cosine():
    with pytest.raises(ValueError):
        cosine_similarity([0, 0], [1, 2])
