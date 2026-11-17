
import pytest

from ..chunker import chunk_text


def test_empty_string_returns_empty_list():
    assert chunk_text("") == []


def test_whitespace_only_string_returns_empty_list():
    assert chunk_text("   \n\t  ") == []


def test_text_shorter_than_chunk_size_returns_one_chunk():
    assert chunk_text(
        "one two three",
        chunk_size=5,
        overlap=0
    ) == [
        "one two three"
    ]


def test_text_equal_to_chunk_size_returns_one_chunk():
    assert chunk_text(
        "one two three four",
        chunk_size=4,
        overlap=0
    ) == [
        "one two three four"
    ]


def test_text_longer_than_chunk_size_is_split():
    assert chunk_text(
        "one two three four five six",
        chunk_size=3,
        overlap=0,
    ) == [
        "one two three",
        "four five six",
    ]


def test_chunks_have_expected_overlap():
    assert chunk_text(
        "one two three four five six seven eight nine ten",
        chunk_size=4,
        overlap=2,
    ) == [
        "one two three four",
        "three four five six",
        "five six seven eight",
        "seven eight nine ten",
    ]


def test_zero_overlap():
    assert chunk_text(
        "one two three four five",
        chunk_size=2,
        overlap=0,
    ) == [
        "one two",
        "three four",
        "five",
    ]


def test_maximum_valid_overlap():
    assert chunk_text(
        "one two three four five",
        chunk_size=3,
        overlap=2,
    ) == [
        "one two three",
        "two three four",
        "three four five",
    ]


@pytest.mark.parametrize("chunk_size", [0, -1, -10])
def test_invalid_chunk_size_values_raise_value_error(chunk_size):
    with pytest.raises(ValueError):
        chunk_text("one two three", chunk_size=chunk_size)


@pytest.mark.parametrize("chunk_size", ["4", 2.5, None, True, False])
def test_invalid_chunk_size_types_raise_type_error(chunk_size):
    with pytest.raises(TypeError):
        chunk_text("one two three", chunk_size=chunk_size)


@pytest.mark.parametrize("overlap", [-1, -5])
def test_negative_overlap_raises_value_error(overlap):
    with pytest.raises(ValueError):
        chunk_text("one two three", chunk_size=3, overlap=overlap)


def test_overlap_equal_to_chunk_size_raises_value_error():
    with pytest.raises(ValueError):
        chunk_text("one two three", chunk_size=3, overlap=3)


def test_overlap_greater_than_chunk_size_raises_value_error():
    with pytest.raises(ValueError):
        chunk_text("one two three", chunk_size=3, overlap=4)


@pytest.mark.parametrize("overlap", ["1", 1.5, None, True, False])
def test_invalid_overlap_types_raise_type_error(overlap):
    with pytest.raises(TypeError):
        chunk_text("one two three", chunk_size=3, overlap=overlap)


@pytest.mark.parametrize("text", [None, 123, ["one", "two"]])
def test_non_string_text_raises_type_error(text):
    with pytest.raises(TypeError):
        chunk_text(text)


def test_whitespace_is_normalized():
    assert chunk_text(
        "one   two\nthree\tfour",
        chunk_size=2,
        overlap=0,
    ) == [
        "one two",
        "three four",
    ]


def test_original_text_is_not_modified():
    text = "one two three four five"
    original_text = text

    chunk_text(text, chunk_size=2, overlap=1)

    assert text == original_text


def test_no_chunk_exceeds_chunk_size():
    text = "one two three four five six seven eight nine"
    chunk_size = 3

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=1)

    assert all(len(chunk.split()) <= chunk_size for chunk in chunks)


def test_reconstructed_word_sequence_preserves_order():
    text = "one two three four five six"
    chunks = chunk_text(text, chunk_size=3, overlap=1)

    # Remove the repeated first word from every chunk after the first.
    reconstructed = chunks[0].split()

    for chunk in chunks[1:]:
        reconstructed.extend(chunk.split()[1:])

    assert reconstructed == text.split()
