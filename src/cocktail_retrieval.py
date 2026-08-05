"""Phase 2 exact-title and semantic retrieval for cocktail documents."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.cocktail_data import normalize_title


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
    device: str | None = None,
) -> SentenceTransformer:
    """Load the Phase 2 English SentenceTransformer embedding model."""
    return SentenceTransformer(model_name, device=device)


def _validate_documents(recipe_documents: Sequence[Mapping[str, Any]]) -> None:
    """Validate the fields needed to create an embedding index."""
    if not recipe_documents:
        raise ValueError("At least one recipe document is required to build an index.")

    required_fields = {"recipe_id", "title", "normalized_title", "document_text"}
    for position, document in enumerate(recipe_documents):
        missing_fields = required_fields - set(document)
        if missing_fields:
            raise KeyError(
                f"Document at position {position} is missing fields: "
                f"{sorted(missing_fields)}"
            )


def generate_document_embeddings(
    recipe_documents: Sequence[Mapping[str, Any]],
    embedding_model: Any,
    *,
    batch_size: int = 32,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """Encode complete documents and L2-normalize their float32 embeddings."""
    _validate_documents(recipe_documents)
    document_texts = [str(document["document_text"]) for document in recipe_documents]
    embeddings = embedding_model.encode(
        document_texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(recipe_documents):
        raise ValueError(
            "Embedding model returned an unexpected shape: " f"{embeddings.shape}"
        )
    faiss.normalize_L2(embeddings)
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create a cosine-similarity FAISS index from L2-normalized vectors."""
    vectors = np.ascontiguousarray(embeddings, dtype=np.float32).copy()
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty two-dimensional array.")

    # Re-normalize defensively so callers cannot accidentally index raw vectors.
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def build_retrieval_bundle(
    recipe_documents: Sequence[Mapping[str, Any]],
    embedding_model: Any,
    *,
    batch_size: int = 32,
    show_progress_bar: bool = True,
) -> dict[str, Any]:
    """Create normalized document embeddings and a matching FAISS IP index."""
    documents = [dict(document) for document in recipe_documents]
    embeddings = generate_document_embeddings(
        documents,
        embedding_model,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    index = build_faiss_index(embeddings)
    return {
        "embedding_model": embedding_model,
        "embedding_model_name": getattr(embedding_model, "model_name", None),
        "documents": documents,
        "embeddings": embeddings,
        "index": index,
    }


def _title_options(known_titles: Iterable[str]) -> list[tuple[str, str]]:
    """Return unique display-title and normalized-title pairs."""
    options: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for title in known_titles:
        display_title = str(title)
        normalized = normalize_title(display_title)
        key = (display_title, normalized)
        if normalized and key not in seen:
            options.append(key)
            seen.add(key)
    return options


def _contains_title_tokens(question_tokens: list[str], title_tokens: list[str]) -> bool:
    """Return whether title tokens appear consecutively in normalized question text."""
    title_length = len(title_tokens)
    if title_length == 0 or title_length > len(question_tokens):
        return False
    return any(
        question_tokens[start : start + title_length] == title_tokens
        for start in range(len(question_tokens) - title_length + 1)
    )


def extract_known_cocktail_title(
    question: str,
    known_titles: Iterable[str],
) -> str | None:
    """Extract the longest known title that appears as full normalized tokens.

    A one-character title is intentionally ignored because it cannot be matched
    reliably inside ordinary English questions.
    """
    normalized_question = normalize_title(question)
    question_tokens = normalized_question.split()
    candidates: list[tuple[int, int, str]] = []

    for display_title, normalized_title in _title_options(known_titles):
        title_tokens = normalized_title.split()
        if len(normalized_title) < 2 or not _contains_title_tokens(
            question_tokens, title_tokens
        ):
            continue
        candidates.append((len(title_tokens), len(normalized_title), display_title))

    if not candidates:
        return None

    # Longest valid title first; display title makes equal candidates stable.
    return sorted(candidates, key=lambda item: (-item[0], -item[1], item[2].casefold()))[
        0
    ][2]


def find_cocktail_by_title(
    cocktail_name: str,
    recipe_documents: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return all case-insensitive exact normalized-title matches.

    Multiple records are returned rather than silently choosing a duplicate
    title. The later assistant layer can surface the ambiguity explicitly.
    """
    normalized_name = normalize_title(cocktail_name)
    if not normalized_name:
        return []
    return [
        dict(document)
        for document in recipe_documents
        if document.get("normalized_title") == normalized_name
    ]


def _format_ingredients(ingredients: Any) -> str:
    """Format preserved source ingredients for structured retrieval evidence."""
    if not isinstance(ingredients, list):
        return str(ingredients or "Not specified")
    formatted = []
    for item in ingredients:
        if not isinstance(item, Mapping):
            continue
        quantity = str(item.get("quantity", "")).strip()
        ingredient = str(item.get("ingredient", "")).strip()
        if ingredient:
            formatted.append(f"{quantity} | {ingredient}" if quantity else ingredient)
    return "; ".join(formatted) or "Not specified"


def _structured_result(
    document: Mapping[str, Any],
    *,
    rank: int,
    similarity: float | None,
    retrieval_method: str,
) -> dict[str, Any]:
    """Build one return value with complete evidence fields."""
    return {
        "rank": rank,
        "recipe_id": document["recipe_id"],
        "source_row_id": document.get("source_row_id"),
        "title": document["title"],
        "similarity": similarity,
        "ingredients": _format_ingredients(document.get("ingredients_original")),
        "document_text": document["document_text"],
        "retrieval_method": retrieval_method,
    }


def retrieve_exact_title(
    question: str,
    recipe_documents: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Extract a known title from a question and return all exact records."""
    extracted_title = extract_known_cocktail_title(
        question, [document["title"] for document in recipe_documents]
    )
    if extracted_title is None:
        return []

    matches = find_cocktail_by_title(extracted_title, recipe_documents)
    return [
        _structured_result(
            document,
            rank=rank,
            similarity=None,
            retrieval_method="exact_title",
        )
        for rank, document in enumerate(matches, start=1)
    ]


def _validate_similarity_threshold(min_similarity: float | None) -> None:
    """Reject thresholds that cannot be cosine similarities."""
    if min_similarity is not None and not -1.0 <= float(min_similarity) <= 1.0:
        raise ValueError("min_similarity must be between -1.0 and 1.0.")


def retrieve_recipes(
    query: str,
    retrieval_bundle: Mapping[str, Any],
    *,
    top_k: int = 5,
    min_similarity: float | None = None,
) -> list[dict[str, Any]]:
    """Run normalized semantic retrieval and return structured evidence results."""
    if not str(query).strip():
        raise ValueError("A non-empty retrieval query is required.")
    if top_k < 1:
        raise ValueError("top_k must be at least one.")
    _validate_similarity_threshold(min_similarity)

    index = retrieval_bundle["index"]
    documents = retrieval_bundle["documents"]
    embedding_model = retrieval_bundle["embedding_model"]
    search_k = min(top_k, len(documents))
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)
    if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
        raise ValueError("Embedding model returned an invalid query embedding shape.")
    faiss.normalize_L2(query_embedding)

    similarities, indices = index.search(query_embedding, search_k)
    results = []
    for rank, (similarity, document_index) in enumerate(
        zip(similarities[0], indices[0]), start=1
    ):
        if document_index < 0:
            continue
        similarity_value = float(similarity)
        if min_similarity is not None and similarity_value < min_similarity:
            continue
        results.append(
            _structured_result(
                documents[int(document_index)],
                rank=rank,
                similarity=similarity_value,
                retrieval_method="semantic",
            )
        )
    return results


def display_retrieval_results(
    results: Sequence[Mapping[str, Any]],
    *,
    preview_characters: int = 260,
) -> pd.DataFrame:
    """Display rank, similarity, evidence, and a shortened document preview."""
    if not results:
        print("No retrieval results met the requested similarity threshold.")
        return pd.DataFrame(
            columns=["rank", "title", "similarity", "ingredients", "document_preview"]
        )

    display_rows = []
    for result in results:
        document_text = str(result["document_text"])
        preview = document_text[:preview_characters].replace("\n", " ")
        if len(document_text) > preview_characters:
            preview += "..."
        display_rows.append(
            {
                "rank": result["rank"],
                "title": result["title"],
                "similarity": result["similarity"],
                "ingredients": result["ingredients"],
                "document_preview": preview,
            }
        )

    display_df = pd.DataFrame(display_rows)
    for row in display_rows:
        score = "exact title" if row["similarity"] is None else f"{row['similarity']:.4f}"
        print(f"{row['rank']}. {row['title']} | similarity: {score}")
        print(f"   Ingredients: {row['ingredients']}")
        print(f"   Preview: {row['document_preview']}")
    return display_df


def evaluate_semantic_retrieval(
    evaluation_cases: Sequence[Mapping[str, str]],
    retrieval_bundle: Mapping[str, Any],
    *,
    top_k: int = 3,
    min_similarity: float | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Calculate preliminary top-1, top-3, and reciprocal-rank retrieval metrics."""
    rows = []
    for case in evaluation_cases:
        query = case["query"]
        expected_title = case["expected_title"]
        expected_normalized_title = normalize_title(expected_title)
        expected_casefold_title = expected_title.casefold()
        results = retrieve_recipes(
            query,
            retrieval_bundle,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        normalized_ranks = [
            result["rank"]
            for result in results
            if normalize_title(result["title"]) == expected_normalized_title
        ]
        strict_ranks = [
            result["rank"]
            for result in results
            if result["title"].casefold() == expected_casefold_title
        ]
        expected_rank = min(strict_ranks) if strict_ranks else None
        normalized_expected_rank = (
            min(normalized_ranks) if normalized_ranks else None
        )
        rows.append(
            {
                "query": query,
                "expected_title": expected_title,
                "top_1_title": results[0]["title"] if results else None,
                "top_1_similarity": results[0]["similarity"] if results else None,
                "top_3_titles": [result["title"] for result in results],
                "strict_expected_rank": expected_rank,
                "normalized_title_family_rank": normalized_expected_rank,
                "top_1_correct": expected_rank == 1,
                "top_3_correct": expected_rank is not None and expected_rank <= 3,
                "reciprocal_rank": 1 / expected_rank if expected_rank else 0.0,
                "normalized_family_top_1_correct": normalized_expected_rank == 1,
                "normalized_family_top_3_correct": (
                    normalized_expected_rank is not None
                    and normalized_expected_rank <= 3
                ),
                "rejected": not results,
            }
        )

    results_df = pd.DataFrame(rows)
    case_count = len(results_df)
    metrics = {
        "case_count": case_count,
        "top_1_accuracy": float(results_df["top_1_correct"].mean()) if case_count else 0.0,
        "top_3_accuracy": float(results_df["top_3_correct"].mean()) if case_count else 0.0,
        "mean_reciprocal_rank": float(results_df["reciprocal_rank"].mean())
        if case_count
        else 0.0,
        "false_retrieval_count": int(
            ((~results_df["top_1_correct"]) & (~results_df["rejected"])).sum()
        )
        if case_count
        else 0,
        "rejected_query_count": int(results_df["rejected"].sum()) if case_count else 0,
        "normalized_family_top_1_accuracy": float(
            results_df["normalized_family_top_1_correct"].mean()
        )
        if case_count
        else 0.0,
        "normalized_family_top_3_accuracy": float(
            results_df["normalized_family_top_3_correct"].mean()
        )
        if case_count
        else 0.0,
    }
    return results_df, metrics


def run_similarity_threshold_experiments(
    queries: Sequence[str],
    retrieval_bundle: Mapping[str, Any],
    *,
    thresholds: Sequence[float] = (0.25, 0.35, 0.45, 0.55, 0.65),
    top_k: int = 3,
) -> pd.DataFrame:
    """Show how threshold changes affect accepted semantic retrieval results."""
    rows = []
    for query in queries:
        unfiltered = retrieve_recipes(query, retrieval_bundle, top_k=top_k)
        top_similarity = unfiltered[0]["similarity"] if unfiltered else None
        top_title = unfiltered[0]["title"] if unfiltered else None
        for threshold in thresholds:
            filtered = retrieve_recipes(
                query,
                retrieval_bundle,
                top_k=top_k,
                min_similarity=threshold,
            )
            rows.append(
                {
                    "query": query,
                    "threshold": float(threshold),
                    "unfiltered_top_title": top_title,
                    "unfiltered_top_similarity": top_similarity,
                    "accepted_result_count": len(filtered),
                    "rejected": not filtered,
                }
            )
    return pd.DataFrame(rows)
