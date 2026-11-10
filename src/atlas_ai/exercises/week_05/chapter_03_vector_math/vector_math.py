import math


def check_length(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimensions")

    if not a or not b:
        raise ValueError("Vectors must not be empty")


def dot_product(a, b):
    check_length(a, b)
    return sum(x * y for x, y in zip(a, b))


def magnitude(vector):
    if not vector:
        raise ValueError("Vector must not be empty")

    return math.hypot(*vector)


def cosine_similarity(a, b):
    check_length(a, b)

    mag_a = magnitude(a)
    mag_b = magnitude(b)

    if mag_a == 0 or mag_b == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors")

    return dot_product(a, b) / (mag_a * mag_b)


def euclidean_distance(a, b):
    check_length(a, b)

    return math.sqrt(
        sum((x - y) ** 2 for x, y in zip(a, b))
    )
