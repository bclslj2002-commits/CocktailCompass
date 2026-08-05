"""Phase 1 data preparation for the cocktail RAG assistant.

The functions in this module intentionally avoid retrieval and language-model
dependencies. They preserve source fields while creating conservative,
deterministic fields that later phases can use for matching and retrieval.
"""

from __future__ import annotations

import ast
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

import pandas as pd


EXPECTED_COLUMNS = ("title", "glass", "garnish", "recipe", "ingredients")
CORE_COLUMNS = ("title", "recipe", "ingredients")

OPTIONAL_INGREDIENT_PATTERN = re.compile(
    r"\b(?:optional|omit\s+if|to\s+taste)\b", re.IGNORECASE
)
PARENTHETICAL_TEXT_PATTERN = re.compile(r"\([^()]*\)")
LEADING_MEASUREMENT_PATTERN = re.compile(
    r"^\s*(?:\d+(?:[.,]\d+)?|one|two|three|four|five|a|an)\s*"
    r"(?:ml|cl|l|oz|dash(?:es)?|drop(?:s)?|slice(?:s)?|sprig(?:s)?|"
    r"whole|cube(?:s)?|spoon(?:s)?|barspoon(?:s)?|part(?:s)?)?(?:[.,]?\s+)",
    re.IGNORECASE,
)

# These normalizations are morphological only. They do not equate different
# ingredients or infer a spirit type from a brand name.
PLURAL_TOKEN_REPLACEMENTS = {
    "berries": "berry",
    "leaves": "leaf",
    "limes": "lime",
    "lemons": "lemon",
    "oranges": "orange",
    "wedges": "wedge",
    "slices": "slice",
    "sprigs": "sprig",
    "cubes": "cube",
    "dashes": "dash",
    "drops": "drop",
}
INGREDIENT_PHRASE_REPLACEMENTS = {
    "mint leaf": "mint",
    "mint leaves": "mint",
}


def is_missing_value(value: Any) -> bool:
    """Return whether a scalar value represents a missing source value."""
    if value is None:
        return True
    if isinstance(value, str):
        return False
    try:
        missing = pd.isna(value)
        return bool(missing) if not hasattr(missing, "__len__") else False
    except (TypeError, ValueError):
        return False


def clean_text(value: Any) -> str:
    """Convert a source value to trimmed, single-space text without inventing data."""
    if is_missing_value(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _ascii_lower_text(value: Any) -> str:
    """Apply case and Unicode normalization used by title and ingredient helpers."""
    text = clean_text(value).lower().replace("&", " and ")
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def normalize_title(title: Any) -> str:
    """Return a stable title key for deterministic, case-insensitive matching.

    Bracketed descriptors and punctuation are removed from the matching key but
    never from the display title retained in the source record.
    """
    normalized = _ascii_lower_text(title)
    normalized = PARENTHETICAL_TEXT_PATTERN.sub(" ", normalized)
    normalized = re.sub(r"\[[^\]]*\]", " ", normalized)
    # Keep dotted initialisms such as "A.B.C." together so that a user query
    # written as "ABC" receives the same normalized title key.
    normalized = re.sub(r"(?<=[a-z])\.(?=[a-z])", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _singularize_known_token(token: str) -> str:
    """Singularize only explicit, reviewable plural forms."""
    return PLURAL_TOKEN_REPLACEMENTS.get(token, token)


def normalize_ingredient_name(ingredient_name: Any) -> str:
    """Create a conservative matching key for a source ingredient name.

    The function removes preparation notes, leading measurements, punctuation,
    and selected plural forms. It deliberately leaves brand names and spirit
    subtypes intact; aliases belong to the later recommendation phase.
    """
    normalized = _ascii_lower_text(ingredient_name)
    normalized = PARENTHETICAL_TEXT_PATTERN.sub(" ", normalized)
    normalized = LEADING_MEASUREMENT_PATTERN.sub("", normalized)
    normalized = re.sub(r"\boptional\b", " ", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if not normalized:
        return ""

    normalized = " ".join(
        _singularize_known_token(token) for token in normalized.split()
    )
    return INGREDIENT_PHRASE_REPLACEMENTS.get(normalized, normalized)


def is_optional_ingredient(ingredient_name: Any) -> bool:
    """Identify only explicit optional wording found in the source ingredient."""
    return bool(OPTIONAL_INGREDIENT_PATTERN.search(clean_text(ingredient_name)))


def parse_ingredients(ingredients_value: Any) -> dict[str, Any]:
    """Safely parse nested source ingredients into structured dictionaries.

    ``ast.literal_eval`` is used for the dataset's Python-style list strings;
    executable ``eval`` is never used. Recoverable one-item or text entries are
    retained with an empty quantity and reported as repairs. Invalid entries are
    counted so they can be shown in the cleaning summary.
    """
    if is_missing_value(ingredients_value) or not clean_text(ingredients_value):
        return {
            "items": [],
            "status": "empty",
            "repair_count": 0,
            "malformed_item_count": 0,
            "message": "Ingredient field is empty.",
        }

    if isinstance(ingredients_value, str):
        try:
            raw_ingredients = ast.literal_eval(ingredients_value)
        except (SyntaxError, ValueError, MemoryError) as error:
            return {
                "items": [],
                "status": "malformed",
                "repair_count": 0,
                "malformed_item_count": 1,
                "message": f"literal_eval failed: {error.__class__.__name__}",
            }
    elif isinstance(ingredients_value, (list, tuple)):
        raw_ingredients = ingredients_value
    else:
        return {
            "items": [],
            "status": "malformed",
            "repair_count": 0,
            "malformed_item_count": 1,
            "message": "Ingredient field is not text, a list, or a tuple.",
        }

    if not isinstance(raw_ingredients, (list, tuple)):
        return {
            "items": [],
            "status": "malformed",
            "repair_count": 0,
            "malformed_item_count": 1,
            "message": "Parsed ingredient field is not a list or tuple.",
        }

    parsed_items: list[dict[str, Any]] = []
    repairs = 0
    malformed_items = 0

    for raw_item in raw_ingredients:
        quantity: Any = ""
        ingredient_name: Any = ""

        if isinstance(raw_item, Mapping):
            quantity = raw_item.get("quantity", raw_item.get("amount", ""))
            ingredient_name = raw_item.get("ingredient", raw_item.get("name", ""))
            repairs += 1
        elif isinstance(raw_item, (list, tuple)):
            if len(raw_item) >= 2:
                quantity, ingredient_name = raw_item[0], raw_item[1]
                if len(raw_item) > 2:
                    repairs += 1
            elif len(raw_item) == 1:
                ingredient_name = raw_item[0]
                repairs += 1
            else:
                malformed_items += 1
                continue
        elif isinstance(raw_item, str):
            ingredient_name = raw_item
            repairs += 1
        else:
            malformed_items += 1
            continue

        original_name = clean_text(ingredient_name)
        normalized_name = normalize_ingredient_name(original_name)
        if not original_name or not normalized_name:
            malformed_items += 1
            continue

        parsed_items.append(
            {
                "quantity": clean_text(quantity),
                "ingredient": original_name,
                "normalized_ingredient": normalized_name,
                "is_optional": is_optional_ingredient(original_name),
            }
        )

    if not parsed_items:
        status = "empty" if not raw_ingredients else "malformed"
    elif repairs or malformed_items:
        status = "repaired"
    else:
        status = "parsed"

    return {
        "items": parsed_items,
        "status": status,
        "repair_count": repairs,
        "malformed_item_count": malformed_items,
        "message": "",
    }


def unique_in_order(values: list[str]) -> list[str]:
    """Return non-empty strings once each while preserving source order."""
    return list(dict.fromkeys(value for value in values if value))


def split_required_and_optional_ingredients(
    ingredient_items: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Split normalized items using only explicit source optional markers."""
    required = [
        item["normalized_ingredient"]
        for item in ingredient_items
        if not item["is_optional"]
    ]
    optional = [
        item["normalized_ingredient"]
        for item in ingredient_items
        if item["is_optional"]
    ]
    return unique_in_order(required), unique_in_order(optional)


def build_recipe_document(recipe: Mapping[str, Any]) -> str:
    """Build one complete, traceable document for a cocktail source record."""
    ingredient_lines = []
    for item in recipe["ingredients_original"]:
        quantity = item["quantity"]
        ingredient = item["ingredient"]
        ingredient_lines.append(
            f"- {quantity} | {ingredient}" if quantity else f"- {ingredient}"
        )

    ingredients_text = "\n".join(ingredient_lines) or "- Not specified"
    glass = recipe["glass"] or "Not specified"
    garnish = recipe["garnish"] or "Not specified"
    return (
        f"Title: {recipe['title']}\n"
        f"Glass: {glass}\n"
        f"Garnish: {garnish}\n\n"
        f"Ingredients:\n{ingredients_text}\n\n"
        f"Preparation:\n{recipe['recipe']}"
    )


def raw_data_quality_summary(raw_df: pd.DataFrame) -> dict[str, Any]:
    """Return visible schema, missing-value, and duplicate summaries for raw data."""
    validate_schema(raw_df)
    missing_rows = []
    for column_name in EXPECTED_COLUMNS:
        null_count = int(raw_df[column_name].map(is_missing_value).sum())
        blank_count = int(
            raw_df[column_name].map(
                lambda value: isinstance(value, str) and not value.strip()
            ).sum()
        )
        missing_rows.append(
            {
                "column": column_name,
                "null_values": null_count,
                "blank_text_values": blank_count,
                "total_missing": null_count + blank_count,
            }
        )

    return {
        "row_count": len(raw_df),
        "column_names": list(raw_df.columns),
        "data_types": raw_df.dtypes.astype(str).to_dict(),
        "missing_values": pd.DataFrame(missing_rows),
        "duplicate_title_count": int(raw_df.duplicated(subset=["title"]).sum()),
        "duplicate_record_count": int(raw_df.duplicated().sum()),
        "duplicate_recipe_ingredient_count": int(
            raw_df.duplicated(subset=["recipe", "ingredients"]).sum()
        ),
    }


def validate_schema(raw_df: pd.DataFrame) -> None:
    """Raise a clear error if a loaded dataset lacks required source fields."""
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(raw_df.columns))
    if missing_columns:
        raise KeyError(
            "The cocktail dataset is missing required columns: "
            f"{missing_columns}. Available columns: {list(raw_df.columns)}"
        )


def prepare_cocktail_records(raw_df: pd.DataFrame) -> dict[str, Any]:
    """Clean source rows and build complete documents without deleting variants.

    Rows lacking title, recipe, or parseable ingredients are excluded from the
    indexed-record output. Missing glass and garnish values are retained as
    blank source values and rendered as ``Not specified`` in documents. Exact
    duplicate rows are removed; duplicate titles and recipe variations remain
    traceable through ``source_row_id``.
    """
    validate_schema(raw_df)
    quality = raw_data_quality_summary(raw_df)

    working_df = raw_df.loc[:, EXPECTED_COLUMNS].copy().reset_index(drop=True)
    working_df.insert(0, "source_row_id", working_df.index.astype(int))
    for column_name in EXPECTED_COLUMNS:
        working_df[column_name] = working_df[column_name].map(clean_text)

    raw_rows = len(working_df)
    missing_by_column = {
        column_name: int((working_df[column_name] == "").sum())
        for column_name in CORE_COLUMNS
    }
    missing_core_mask = working_df.loc[:, CORE_COLUMNS].eq("").any(axis=1)
    rows_removed_missing_core = int(missing_core_mask.sum())
    working_df = working_df.loc[~missing_core_mask].copy()

    rows_before_deduplication = len(working_df)
    working_df = working_df.drop_duplicates(
        subset=list(EXPECTED_COLUMNS), keep="first"
    ).copy()
    rows_removed_exact_duplicates = rows_before_deduplication - len(working_df)

    parse_results = working_df["ingredients"].map(parse_ingredients)
    working_df["ingredient_parse_status"] = parse_results.map(
        lambda result: result["status"]
    )
    working_df["ingredient_parse_message"] = parse_results.map(
        lambda result: result["message"]
    )
    working_df["ingredient_repair_count"] = parse_results.map(
        lambda result: result["repair_count"]
    )
    working_df["malformed_ingredient_item_count"] = parse_results.map(
        lambda result: result["malformed_item_count"]
    )
    working_df["ingredients_original"] = parse_results.map(
        lambda result: result["items"]
    )

    valid_ingredient_mask = working_df["ingredients_original"].map(bool)
    rows_removed_invalid_ingredients = int((~valid_ingredient_mask).sum())
    invalid_status_counts = (
        working_df.loc[~valid_ingredient_mask, "ingredient_parse_status"]
        .value_counts()
        .to_dict()
    )
    working_df = working_df.loc[valid_ingredient_mask].copy()

    working_df["normalized_title"] = working_df["title"].map(normalize_title)
    working_df["required_ingredients"] = working_df["ingredients_original"].map(
        lambda items: split_required_and_optional_ingredients(items)[0]
    )
    working_df["optional_ingredients"] = working_df["ingredients_original"].map(
        lambda items: split_required_and_optional_ingredients(items)[1]
    )
    working_df["normalized_ingredients"] = working_df.apply(
        lambda row: unique_in_order(
            row["required_ingredients"] + row["optional_ingredients"]
        ),
        axis=1,
    )
    working_df["recipe_id"] = range(len(working_df))

    records = []
    for row in working_df.to_dict(orient="records"):
        record = {
            "recipe_id": int(row["recipe_id"]),
            "source_row_id": int(row["source_row_id"]),
            "title": row["title"],
            "normalized_title": row["normalized_title"],
            "glass": row["glass"],
            "garnish": row["garnish"],
            "recipe": row["recipe"],
            "ingredients_raw": row["ingredients"],
            "ingredients_original": row["ingredients_original"],
            "required_ingredients": row["required_ingredients"],
            "optional_ingredients": row["optional_ingredients"],
            "normalized_ingredients": row["normalized_ingredients"],
            "ingredient_parse_status": row["ingredient_parse_status"],
        }
        record["document_text"] = build_recipe_document(record)
        records.append(record)

    record_documents = {record["recipe_id"]: record["document_text"] for record in records}
    working_df["document_text"] = working_df["recipe_id"].map(record_documents)
    retained_duplicate_title_count = int(working_df.duplicated(subset=["title"]).sum())
    retained_normalized_title_duplicate_count = int(
        working_df.duplicated(subset=["normalized_title"]).sum()
    )
    retained_recipe_ingredient_duplicate_count = int(
        working_df.duplicated(subset=["recipe", "ingredients"]).sum()
    )
    summary_rows = [
        {"step": "Raw source rows", "count": raw_rows},
        {
            "step": "Rows removed: missing title, recipe, or ingredients",
            "count": rows_removed_missing_core,
        },
        {
            "step": "Rows removed: exact duplicate source records",
            "count": rows_removed_exact_duplicates,
        },
        {
            "step": "Rows removed: empty or malformed parsed ingredients",
            "count": rows_removed_invalid_ingredients,
        },
        {
            "step": "Repaired ingredient entries retained",
            "count": int(working_df["ingredient_repair_count"].sum()),
        },
        {"step": "Retained complete recipe documents", "count": len(records)},
        {
            "step": "Duplicate source titles retained for traceability",
            "count": retained_duplicate_title_count,
        },
        {
            "step": "Normalized-title collisions retained for later disambiguation",
            "count": retained_normalized_title_duplicate_count,
        },
        {
            "step": "Duplicate recipe-and-ingredient combinations retained",
            "count": retained_recipe_ingredient_duplicate_count,
        },
    ]

    return {
        "records": records,
        "cleaned_dataframe": working_df.reset_index(drop=True),
        "data_quality": quality,
        "cleaning_summary": pd.DataFrame(summary_rows),
        "details": {
            "missing_core_by_column": missing_by_column,
            "invalid_ingredient_status_counts": invalid_status_counts,
            "retained_records": len(records),
            "duplicate_title_count_retained": retained_duplicate_title_count,
            "normalized_title_duplicate_count_retained": retained_normalized_title_duplicate_count,
            "duplicate_recipe_ingredient_count_retained": retained_recipe_ingredient_duplicate_count,
        },
    }


def run_data_preparation_self_tests() -> dict[str, Any]:
    """Run small deterministic tests that do not require dataset downloads."""
    assert normalize_title(" A.B.C. Cocktail (classic) ") == "abc cocktail"
    assert normalize_ingredient_name("Lime Juice (Freshly Squeezed)") == "lime juice"
    assert normalize_ingredient_name("2 oz. Mint Leaves") == "mint"

    parsed = parse_ingredients(
        "[['50 ml', 'White Rum'], ['2 dash', 'Bitters (optional)']]"
    )
    assert parsed["status"] == "parsed"
    assert parsed["items"][0]["normalized_ingredient"] == "white rum"
    assert parsed["items"][1]["is_optional"] is True

    repaired = parse_ingredients(["Mint leaves", ["25 ml", "Lime juice"]])
    assert repaired["status"] == "repaired"
    assert repaired["items"][0]["quantity"] == ""
    assert repaired["items"][0]["normalized_ingredient"] == "mint"

    malformed = parse_ingredients("not a valid ingredient list")
    assert malformed["status"] == "malformed"
    assert malformed["items"] == []

    fixture = pd.DataFrame(
        [
            {
                "title": "Test Drink",
                "glass": "Highball glass",
                "garnish": None,
                "recipe": "STIR with ice.",
                "ingredients": "[['50 ml', 'Gin'], ['1 dash', 'Bitters (optional)']]",
            },
            {
                "title": "Test Drink",
                "glass": "Highball glass",
                "garnish": None,
                "recipe": "STIR with ice.",
                "ingredients": "[['50 ml', 'Gin'], ['1 dash', 'Bitters (optional)']]",
            },
            {
                "title": "No Recipe",
                "glass": "",
                "garnish": "",
                "recipe": None,
                "ingredients": "[['50 ml', 'Gin']]",
            },
            {
                "title": "Bad Ingredients",
                "glass": "Coupe glass",
                "garnish": "",
                "recipe": "SHAKE.",
                "ingredients": "not valid",
            },
        ]
    )
    prepared = prepare_cocktail_records(fixture)
    assert prepared["details"]["retained_records"] == 1
    record = prepared["records"][0]
    assert record["required_ingredients"] == ["gin"]
    assert record["optional_ingredients"] == ["bitters"]
    assert "Garnish: Not specified" in record["document_text"]
    assert "50 ml | Gin" in record["document_text"]

    return {
        "status": "passed",
        "tests": [
            "title normalization",
            "ingredient normalization",
            "safe parsing",
            "repair handling",
            "malformed-input handling",
            "cleaning and document construction",
        ],
    }
