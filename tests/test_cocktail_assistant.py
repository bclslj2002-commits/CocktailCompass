"""Tests for deterministic assistant routing, matching, and grounded fallbacks."""

import numpy as np

from src.cocktail_assistant import (
    INGREDIENT_RECOMMENDATION,
    OUT_OF_SCOPE,
    RECIPE_LOOKUP,
    SEMANTIC_RAG_RECOMMENDATION,
    ask_cocktail_assistant,
    classify_intent,
    format_recipe_answer,
    parse_user_ingredients,
    recommend_by_ingredients,
    score_ingredient_match,
)
from src.cocktail_data import normalize_title
from src.cocktail_retrieval import build_retrieval_bundle


class FakeEmbeddingModel:
    """A local deterministic encoder; language-model downloads are not needed."""

    def encode(self, texts, **_kwargs):
        vectors = {
            "gin cooler document": [1.0, 0.0],
            "creamy night document": [0.0, 1.0],
            "How do I make a fictional Blue Moon Dragon?": [-1.0, -1.0],
            "How do I make a fictional cocktail called Blue Moon Dragon?": [-1.0, -1.0],
            "Find a light gin cocktail for summer.": [1.0, 0.0],
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)


def _documents() -> list[dict]:
    return [
        {
            "recipe_id": 1,
            "source_row_id": 1,
            "title": "Gin Cooler",
            "normalized_title": normalize_title("Gin Cooler"),
            "glass": "Highball glass",
            "garnish": "Mint sprig",
            "recipe": "Build with ice and top with soda water.",
            "ingredients_original": [
                {"quantity": "50 ml", "ingredient": "Gin"},
                {"quantity": "25 ml", "ingredient": "Lemon juice"},
                {"quantity": "100 ml", "ingredient": "Soda water"},
                {"quantity": "6", "ingredient": "Mint leaves"},
            ],
            "required_ingredients": ["gin", "lemon juice", "soda water", "mint"],
            "optional_ingredients": [],
            "normalized_ingredients": ["gin", "lemon juice", "soda water", "mint"],
            "document_text": "gin cooler document",
        },
        {
            "recipe_id": 2,
            "source_row_id": 2,
            "title": "Creamy Night",
            "normalized_title": normalize_title("Creamy Night"),
            "glass": "Coupe",
            "garnish": "Not specified",
            "recipe": "Shake and strain.",
            "ingredients_original": [
                {"quantity": "40 ml", "ingredient": "Coffee liqueur"},
                {"quantity": "40 ml", "ingredient": "Cream"},
            ],
            "required_ingredients": ["coffee liqueur", "cream"],
            "optional_ingredients": [],
            "normalized_ingredients": ["coffee liqueur", "cream"],
            "document_text": "creamy night document",
        },
    ]


def _bundle() -> dict:
    return build_retrieval_bundle(_documents(), FakeEmbeddingModel(), show_progress_bar=False)


def test_ingredient_aliases_and_missing_ingredients_are_deterministic() -> None:
    documents = _documents()
    assert parse_user_ingredients(["Gin", "Lemon juice", "club soda", "Mint leaves"]) == [
        "gin",
        "lemon juice",
        "soda water",
        "mint",
    ]
    score = score_ingredient_match(documents[0], ["gin", "lemon juice"])
    assert score["matched_required_count"] == 2
    assert score["missing_required_ingredients"] == ["mint", "soda water"]
    assert score["match_ratio"] == 0.5


def test_ingredient_recommendation_ranking_and_source_formatting() -> None:
    results = recommend_by_ingredients(
        ["gin", "lemon juice", "club soda", "mint"], _documents(), top_n=2
    )
    assert results[0]["title"] == "Gin Cooler"
    assert results[0]["can_make_now"] is True
    source_answer = format_recipe_answer(_documents()[0])
    assert "50 ml | Gin" in source_answer
    assert "Mint sprig" in source_answer


def test_preference_recommendation_uses_faiss_and_visible_source_evidence() -> None:
    result = ask_cocktail_assistant(
        "Find a light gin cocktail for summer.",
        _documents(),
        _bundle(),
        generate_answer=False,
    )
    assert result["intent"] == SEMANTIC_RAG_RECOMMENDATION
    assert result["retrieved_recipes"][0]["retrieval_method"] == "semantic"
    assert result["retrieved_recipes"][0]["title"] == "Gin Cooler"
    assert any(
        "Embedding + FAISS semantic retrieval was used" in warning
        for warning in result["warnings"]
    )
    assert "50 ml | Gin" in result["answer"]


def test_intent_router_prefers_explicit_inputs_and_rejects_out_of_scope() -> None:
    assert classify_intent("How do I make a Gin Cooler?") == RECIPE_LOOKUP
    assert classify_intent("Anything", user_ingredients=["gin"]) == INGREDIENT_RECOMMENDATION
    assert (
        classify_intent("Anything", user_ingredients=["gin"], requested_style="refreshing")
        == SEMANTIC_RAG_RECOMMENDATION
    )
    assert classify_intent("Recommend a cocktail with ginger and lime.") == SEMANTIC_RAG_RECOMMENDATION
    assert classify_intent("Who wrote Hamlet?") == OUT_OF_SCOPE


def test_assistant_exact_title_fictional_rejection_and_out_of_scope() -> None:
    documents = _documents()
    exact = ask_cocktail_assistant(
        "How do I make a GIN COOLER?",
        documents,
        _bundle(),
        generate_answer=False,
    )
    assert exact["intent"] == RECIPE_LOOKUP
    assert exact["retrieved_recipes"][0]["retrieval_method"] == "exact_title"
    assert "50 ml | Gin" in exact["answer"]

    fictional = ask_cocktail_assistant(
        "How do I make a fictional Blue Moon Dragon?",
        documents,
        _bundle(),
        generate_answer=False,
    )
    assert fictional["retrieved_recipes"] == []
    assert "could not find a sufficiently reliable recipe" in fictional["answer"]

    unrelated = ask_cocktail_assistant("Who wrote Hamlet?", documents, _bundle())
    assert unrelated["intent"] == OUT_OF_SCOPE
    assert "available-ingredient matching" in unrelated["answer"]


def test_explicit_cocktail_suffix_supports_a_parent_name_without_fuzzy_matching() -> None:
    documents = _documents()
    documents[0]["title"] = "Gin Cooler Cocktail"
    documents[0]["normalized_title"] = normalize_title(documents[0]["title"])
    outcome = ask_cocktail_assistant(
        "How do I make a Gin Cooler?", documents, _bundle(), generate_answer=False
    )
    assert outcome["retrieved_recipes"][0]["retrieval_method"] == "canonical_title_alias"
    assert outcome["retrieved_recipes"][0]["title"] == "Gin Cooler Cocktail"


def test_canonical_alias_never_overrides_an_explicit_fictional_name() -> None:
    documents = _documents()
    documents[0]["title"] = "Blue Moon Cocktail"
    documents[0]["normalized_title"] = normalize_title(documents[0]["title"])
    outcome = ask_cocktail_assistant(
        "How do I make a fictional cocktail called Blue Moon Dragon?",
        documents,
        _bundle(),
        generate_answer=False,
    )
    assert outcome["retrieved_recipes"] == []


def test_literal_title_wins_over_a_shorter_canonical_parent_alias() -> None:
    documents = _documents()
    documents[0]["title"] = "Gin Cocktail"
    documents[0]["normalized_title"] = normalize_title(documents[0]["title"])
    documents[1]["title"] = "Gin Fizz"
    documents[1]["normalized_title"] = normalize_title(documents[1]["title"])
    outcome = ask_cocktail_assistant(
        "What ingredients are needed for a Gin Fizz?",
        documents,
        _bundle(),
        generate_answer=False,
    )
    assert outcome["retrieved_recipes"][0]["title"] == "Gin Fizz"
