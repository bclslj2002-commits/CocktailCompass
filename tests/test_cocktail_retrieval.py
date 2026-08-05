"""Tests for Phase 2 title matching and FAISS retrieval."""

import numpy as np

from src.cocktail_data import normalize_title
from src.cocktail_retrieval import (
    build_retrieval_bundle,
    evaluate_semantic_retrieval,
    extract_known_cocktail_title,
    find_cocktail_by_title,
    retrieve_recipes,
)


class FakeEmbeddingModel:
    """Small deterministic encoder used to test FAISS without model downloads."""

    def __init__(self) -> None:
        self.vectors = {
            "tom document": [1.0, 0.0],
            "espresso document": [0.0, 1.0],
            "a tall gin drink with lemon": [0.98, 0.2],
            "fictional celestial dragon beverage": [0.1, 0.1],
        }

    def encode(self, texts, **_kwargs):
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def _fixture_documents() -> list[dict]:
    return [
        {
            "recipe_id": 1,
            "source_row_id": 10,
            "title": "Tom Collins",
            "normalized_title": normalize_title("Tom Collins"),
            "ingredients_original": [{"quantity": "6 cl", "ingredient": "Gin"}],
            "document_text": "tom document",
        },
        {
            "recipe_id": 2,
            "source_row_id": 11,
            "title": "Espresso Martini",
            "normalized_title": normalize_title("Espresso Martini"),
            "ingredients_original": [{"quantity": "3 cl", "ingredient": "Espresso"}],
            "document_text": "espresso document",
        },
    ]


def _fixture_bundle() -> dict:
    return build_retrieval_bundle(
        _fixture_documents(), FakeEmbeddingModel(), show_progress_bar=False
    )


def test_exact_title_matching_handles_case_and_natural_question() -> None:
    titles = ["Tom Collins", "Espresso Martini"]
    assert extract_known_cocktail_title("How do I make a TOM COLLINS?", titles) == "Tom Collins"
    assert (
        extract_known_cocktail_title(
            "Please give me the glass and ingredients for an Espresso Martini.", titles
        )
        == "Espresso Martini"
    )


def test_find_cocktail_by_title_is_case_insensitive() -> None:
    matches = find_cocktail_by_title("eSpReSsO mArTiNi", _fixture_documents())
    assert len(matches) == 1
    assert matches[0]["recipe_id"] == 2


def test_semantic_query_returns_structured_top_match() -> None:
    results = retrieve_recipes(
        "a tall gin drink with lemon", _fixture_bundle(), top_k=2
    )
    assert results[0]["title"] == "Tom Collins"
    assert results[0]["similarity"] > 0.9
    assert set(results[0]) >= {
        "rank",
        "recipe_id",
        "title",
        "similarity",
        "ingredients",
        "document_text",
    }


def test_fictional_cocktail_has_no_exact_title_and_weak_match_is_rejected() -> None:
    question = "How do I make a fictional celestial dragon beverage?"
    assert extract_known_cocktail_title(question, ["Tom Collins", "Espresso Martini"]) is None
    results = retrieve_recipes(
        "fictional celestial dragon beverage",
        _fixture_bundle(),
        top_k=2,
        min_similarity=0.8,
    )
    assert results == []


def test_evaluation_reports_strict_and_normalized_title_metrics() -> None:
    evaluation_df, metrics = evaluate_semantic_retrieval(
        [
            {
                "query": "a tall gin drink with lemon",
                "expected_title": "Tom Collins",
            }
        ],
        _fixture_bundle(),
    )
    assert evaluation_df.loc[0, "strict_expected_rank"] == 1
    assert evaluation_df.loc[0, "normalized_title_family_rank"] == 1
    assert metrics["top_1_accuracy"] == 1.0
