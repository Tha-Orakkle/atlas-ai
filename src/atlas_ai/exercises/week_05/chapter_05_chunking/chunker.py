# Chapter 5 Practical Exercise
# Implement a basic chunking utility independently.
# For this exercise:
# - Treat whitespace-separated words as units.
# - chunk_size is the maximum number of words per chunk.
# - overlap is the number of words shared between consecutive chunks.
# - Preserve the original word order.
# - Reject invalid chunk sizes and overlap values.
# - Handle empty or whitespace-only input.
# - Do not modify the original input.

def _validate_input(text, chunk_size, overlap):
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError("chunk_size must be an integer")

    if not isinstance(overlap, int) or isinstance(overlap, bool):
        raise TypeError("overlap must be an integer")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap must not be a negative number")

    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")


def chunk_text(text, chunk_size=100, overlap=20):
    _validate_input(text, chunk_size, overlap)

    splitted_text = text.split()
    chunks = []

    if len(splitted_text) == 0:
        return chunks

    steps = chunk_size - overlap
    text_len = len(splitted_text)

    for i in range(0, text_len, steps):
        end = min(
            text_len,
            i + chunk_size
        )
        chunks.append(
            " ".join(splitted_text[i: end])
        )
        if end == text_len:
            break

    return chunks


text = "one two three four five six seven eight nine ten"
chunk_size = 4
overlap = 2

print(chunk_text(
    text=text,
    chunk_size=chunk_size,
    overlap=overlap
))
