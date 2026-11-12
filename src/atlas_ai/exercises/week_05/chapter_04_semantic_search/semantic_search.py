from atlas_ai.exercises.week_05.chapter_03_vector_math.vector_math import (
    cosine_similarity,
)


class SemanticSearchEngine:

    def __init__(self):
        self.documents = []

    def add_document(self, document, embedding):
        document_id = len(self.documents) + 1
        self.documents.append({
            "id": document_id,
            "text": document,
            "embedding": embedding
        })

    def search(self, query_embedding, top_k=3):
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer")

        if top_k < 0:
            raise ValueError("top_k cannot be negative")

        results = []

        for document in self.documents:
            score = cosine_similarity(
                query_embedding,
                document["embedding"]
            )

            results.append({
                "id": document["id"],
                "text": document["text"],
                "score": score,
            })

        return sorted(
            results,
            key=lambda result: result["score"],
            reverse=True
        )[:top_k]
