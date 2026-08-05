"""Tests for Phase 1 data-preparation functions."""

import pandas as pd

from src.cocktail_data import (
    normalize_ingredient_name,
    normalize_title,
    parse_ingredients,
    prepare_cocktail_records,
    run_data_preparation_self_tests,
)


def test_self_test_suite_passes() -> None:
    assert run_data_preparation_self_tests()["status"] == "passed"


def test_normalization_preserves_brand_identity() -> None:
    assert normalize_title("Alaska (Savoy recipe)") == "alaska"
    assert (
        normalize_ingredient_name("Havana Club 3 Year Old rum")
        == "havana club 3 year old rum"
    )


def test_parser_never_uses_executable_evaluation() -> None:
    result = parse_ingredients("__import__('os').system('echo unsafe')")
    assert result["status"] == "malformed"
    assert result["items"] == []


def test_duplicate_titles_are_retained_but_exact_duplicates_are_removed() -> None:
    frame = pd.DataFrame(
        [
            {
                "title": "Variant",
                "glass": "Coupe",
                "garnish": "",
                "recipe": "STIR.",
                "ingredients": "[['1 cl', 'Gin']]",
            },
            {
                "title": "Variant",
                "glass": "Coupe",
                "garnish": "",
                "recipe": "SHAKE.",
                "ingredients": "[['1 cl', 'Gin']]",
            },
            {
                "title": "Variant",
                "glass": "Coupe",
                "garnish": "",
                "recipe": "SHAKE.",
                "ingredients": "[['1 cl', 'Gin']]",
            },
        ]
    )

    prepared = prepare_cocktail_records(frame)

    assert prepared["details"]["retained_records"] == 2
    assert [record["source_row_id"] for record in prepared["records"]] == [0, 1]
