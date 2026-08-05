"""Grounded local cocktail assistant and deterministic recommendation helpers.

The module keeps recipe facts and ingredient-match decisions in ordinary
Python data structures. Natural-language preference questions use embedding
retrieval with FAISS, then pass only the retrieved recipe evidence to the
local language model for a grounded framing sentence. Factual recipe fields
remain source-rendered so the model cannot change quantities or methods.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

import pandas as pd

from src.cocktail_data import normalize_ingredient_name, normalize_title, unique_in_order
from src.cocktail_retrieval import (
    display_retrieval_results,
    retrieve_exact_title,
    retrieve_recipes,
)


PRIMARY_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
COMPATIBILITY_FALLBACK_MODEL_NAME = "microsoft/Phi-4-mini-instruct"
MIN_RETRIEVAL_SIMILARITY = 0.65
PREFERENCE_MIN_RETRIEVAL_SIMILARITY = 0.20
MAX_NEW_TOKENS = 300

RECIPE_LOOKUP = "RECIPE_LOOKUP"
INGREDIENT_RECOMMENDATION = "INGREDIENT_RECOMMENDATION"
SEMANTIC_RAG_RECOMMENDATION = "SEMANTIC_RAG_RECOMMENDATION"
OUT_OF_SCOPE = "OUT_OF_SCOPE"

# Deliberately small and reviewable aliases.  These express spelling or
# commonly stated equivalents only; they never equate different spirit types.
INGREDIENT_ALIASES = {
    "light rum": "white rum",
    "simple syrup": "sugar syrup",
    "club soda": "soda water",
}
DEFAULT_PANTRY_INGREDIENTS = frozenset({"ice", "ice cube", "water", "plain water"})

GROUNDING_SYSTEM_PROMPT = """You are a careful English-only cocktail assistant.
Use only supplied cocktail records and deterministic results. Do not add or
change ingredients, quantities, garnish, preparation, ingredient-match
calculations, missing ingredients, or semantic retrieval rankings. Do not
answer unrelated questions or claim health benefits from alcohol. Include a
responsible-drinking note where appropriate. If evidence is insufficient, say
so. The factual answer is rendered separately from source data."""


def inspect_runtime() -> dict[str, Any]:
    """Return the local PyTorch/CUDA runtime state without loading a model."""
    import torch

    cuda_available = bool(torch.cuda.is_available())
    gpu_names = (
        [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if cuda_available
        else []
    )
    memory_mib = (
        [
            int(torch.cuda.get_device_properties(index).total_memory / (1024**2))
            for index in range(torch.cuda.device_count())
        ]
        if cuda_available
        else []
    )
    return {
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_count": int(torch.cuda.device_count()),
        "gpu_names": gpu_names,
        "gpu_memory_mib": memory_mib,
        "warning": (
            None
            if cuda_available
            else "CUDA is unavailable. The declared final model should be run on a GPU; CPU generation will be slow."
        ),
    }


def _quantization_config() -> Any:
    """Build the required 4-bit NF4 configuration only when CUDA is available."""
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def _load_quantized_model(model_name: str, cache_dir: str | None = None) -> dict[str, Any]:
    """Load one causal model using the selected CUDA 4-bit configuration."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=_quantization_config(),
        device_map="auto",
    )
    model.eval()
    return {
        "model": model,
        "tokenizer": tokenizer,
        "selected_model": model_name,
        "quantization_method": "4-bit NF4 with double quantization; float16 compute; device_map=auto",
        "device_map": getattr(model, "hf_device_map", None),
        "model_input_device": str(next(model.parameters()).device),
    }


def load_local_language_model(
    *,
    cache_dir: str | None = None,
    allow_compatibility_fallback: bool = True,
) -> dict[str, Any]:
    """Load Qwen in 4-bit CUDA mode and report an explicit fallback outcome.

    Qwen is always attempted first on CUDA.  If it fails, Phi-4-mini-instruct
    is attempted with the same quantization configuration.  On CPU-only
    systems neither large model is silently downloaded or loaded; callers get
    a deterministic-answer-only bundle and a clear warning instead.
    """
    runtime = inspect_runtime()
    base = {
        "model": None,
        "tokenizer": None,
        "selected_model": None,
        "quantization_method": "not loaded",
        "device_map": None,
        "model_input_device": None,
        "runtime": runtime,
        "fallback_used": False,
        "primary_error": None,
        "fallback_error": None,
        "warning": runtime["warning"],
    }
    if not runtime["cuda_available"]:
        return base

    model_errors = (ImportError, OSError, RuntimeError, TypeError, ValueError)
    try:
        loaded = _load_quantized_model(PRIMARY_MODEL_NAME, cache_dir=cache_dir)
        return {**base, **loaded, "warning": None}
    except model_errors as error:
        base["primary_error"] = f"{error.__class__.__name__}: {error}"

    if not allow_compatibility_fallback:
        base["warning"] = "Primary Qwen loading failed and compatibility fallback was disabled."
        return base

    try:
        loaded = _load_quantized_model(COMPATIBILITY_FALLBACK_MODEL_NAME, cache_dir=cache_dir)
        return {
            **base,
            **loaded,
            "fallback_used": True,
            "warning": "Primary Qwen loading failed; the documented Phi-4 compatibility fallback is active.",
        }
    except model_errors as error:
        base["fallback_used"] = True
        base["fallback_error"] = f"{error.__class__.__name__}: {error}"
        base["warning"] = "Both CUDA model-loading attempts failed; deterministic source formatting remains active."
        return base


def _normalise_user_ingredient(value: Any) -> str:
    """Normalize a user ingredient and apply only the explicit alias dictionary."""
    normalized = normalize_ingredient_name(value)
    return INGREDIENT_ALIASES.get(normalized, normalized)


def parse_user_ingredients(text_or_ingredients: str | Sequence[str] | None) -> list[str]:
    """Parse a short English ingredient list into conservative normalized names."""
    if text_or_ingredients is None:
        return []
    if isinstance(text_or_ingredients, str):
        text = re.sub(r"^.*?\b(?:i have|with these ingredients|available ingredients)\b", "", text_or_ingredients, flags=re.I)
        raw_items = re.split(r",|\band\b|;|\n", text)
    else:
        raw_items = [str(item) for item in text_or_ingredients]
    return unique_in_order([_normalise_user_ingredient(item) for item in raw_items])


def get_recipe_ingredient_sets(recipe: Mapping[str, Any]) -> dict[str, set[str]]:
    """Return required and optional normalized ingredient sets for one recipe."""
    return {
        "required": set(recipe.get("required_ingredients", [])),
        "optional": set(recipe.get("optional_ingredients", [])),
    }


def score_ingredient_match(
    recipe: Mapping[str, Any],
    user_ingredients: Sequence[str],
    *,
    pantry_defaults: Sequence[str] = tuple(DEFAULT_PANTRY_INGREDIENTS),
) -> dict[str, Any]:
    """Calculate source-grounded ingredient availability for one recipe."""
    sets = get_recipe_ingredient_sets(recipe)
    available = set(parse_user_ingredients(user_ingredients)) | {
        _normalise_user_ingredient(item) for item in pantry_defaults
    }
    required = sets["required"]
    optional = sets["optional"]
    matched_required = sorted(required & available)
    missing_required = sorted(required - available)
    matched_optional = sorted(optional & available)
    missing_optional = sorted(optional - available)
    required_count = len(required)
    matched_count = len(matched_required)
    return {
        "recipe_id": recipe["recipe_id"],
        "title": recipe["title"],
        "recipe": dict(recipe),
        "required_ingredients": sorted(required),
        "optional_ingredients": sorted(optional),
        "matched_required_ingredients": matched_required,
        "missing_required_ingredients": missing_required,
        "matched_optional_ingredients": matched_optional,
        "missing_optional_ingredients": missing_optional,
        "required_ingredient_count": required_count,
        "matched_required_count": matched_count,
        "missing_count": len(missing_required),
        "match_ratio": matched_count / required_count if required_count else 0.0,
        "can_make_now": len(missing_required) == 0 and required_count > 0,
    }


def recommend_by_ingredients(
    user_ingredients: str | Sequence[str],
    recipe_documents: Sequence[Mapping[str, Any]],
    *,
    top_n: int = 5,
    pantry_defaults: Sequence[str] = tuple(DEFAULT_PANTRY_INGREDIENTS),
) -> list[dict[str, Any]]:
    """Rank recipes with the specified deterministic ingredient-match order."""
    if top_n < 1:
        raise ValueError("top_n must be at least one.")
    normalized_user = parse_user_ingredients(user_ingredients)
    scored = [
        score_ingredient_match(document, normalized_user, pantry_defaults=pantry_defaults)
        for document in recipe_documents
    ]
    scored.sort(
        key=lambda item: (
            item["missing_count"] != 0,
            -item["match_ratio"],
            item["missing_count"],
            item["required_ingredient_count"],
            normalize_title(item["title"]),
            item["recipe_id"],
        )
    )
    for rank, item in enumerate(scored[:top_n], start=1):
        item["rank"] = rank
        item["user_ingredients"] = normalized_user
    return scored[:top_n]


def _format_source_ingredients(recipe: Mapping[str, Any]) -> str:
    """Render quantities and ingredient names directly from preserved source data."""
    lines = []
    for item in recipe.get("ingredients_original", []):
        quantity = str(item.get("quantity", "")).strip()
        ingredient = str(item.get("ingredient", "")).strip()
        if ingredient:
            lines.append(f"- {quantity} | {ingredient}" if quantity else f"- {ingredient}")
    return "\n".join(lines) or "- Not specified in the source record"


def format_recipe_answer(recipe: Mapping[str, Any]) -> str:
    """Render one complete recipe response without any generated factual text."""
    safety_note = ""
    if any(word in str(recipe.get("recipe", "")).casefold() for word in ("flame", "flamb", "ignite")):
        safety_note = "\nSafety: The source preparation mentions flame or ignition; use appropriate fire safety."
    return (
        f"Retrieved source title: {recipe['title']}\n"
        f"Glass: {recipe.get('glass') or 'Not specified'}\n"
        f"Garnish: {recipe.get('garnish') or 'Not specified'}\n\n"
        f"Ingredients:\n{_format_source_ingredients(recipe)}\n\n"
        f"Preparation:\n{recipe.get('recipe') or 'Not specified'}"
        f"{safety_note}\n\nPlease drink responsibly."
    )


def format_ingredient_recommendations(results: Sequence[Mapping[str, Any]]) -> str:
    """Render rankings and missing ingredients directly from deterministic scores."""
    if not results:
        return "No candidate recipes were available for ingredient matching."
    sections = ["Best matches based on your ingredients:"]
    for item in results:
        status = "You can make this cocktail now" if item["can_make_now"] else "Required ingredients are missing"
        missing = ", ".join(item["missing_required_ingredients"]) or "None"
        available = ", ".join(item["matched_required_ingredients"]) or "None"
        optional_missing = ", ".join(item["missing_optional_ingredients"]) or "None"
        sections.append(
            f"\n{item['rank']}. {item['title']}\n"
            f"   Match: {item['matched_required_count']}/{item['required_ingredient_count']} required ingredients\n"
            f"   Status: {status}\n"
            f"   Available required ingredients: {available}\n"
            f"   Missing required ingredients: {missing}\n"
            f"   Optional ingredients not available: {optional_missing}"
        )
    return "\n".join(sections) + "\n\nPlease drink responsibly."


def format_semantic_rag_recommendations(
    results: Sequence[Mapping[str, Any]],
    recipe_documents: Sequence[Mapping[str, Any]],
) -> str:
    """Render the selected FAISS evidence without generated recipe facts.

    The cosine similarity is shown as retrieval evidence, not as a quality or
    flavour score.  Each listed recipe is recovered from its original source
    record, so quantities, ingredients, and methods stay auditable.
    """
    if not results:
        return (
            "I could not find sufficiently relevant cocktail evidence for that "
            "preference in the indexed dataset."
        )

    sections = [
        "Semantic RAG recommendations (retrieved with Embedding + FAISS):",
        "Cosine similarity ranks relevance to your wording; it is not a flavour score.",
    ]
    for result in results:
        recipe = _record_for_result(result, recipe_documents)
        if recipe is None:
            continue
        sections.append(
            f"\n{result['rank']}. {recipe['title']} "
            f"(FAISS cosine similarity: {result['similarity']:.3f})\n"
            f"   Source ingredient evidence: {result['ingredients']}\n"
            f"   Glass: {recipe.get('glass') or 'Not specified'}\n"
            f"   Garnish: {recipe.get('garnish') or 'Not specified'}\n"
            f"   Preparation: {recipe.get('recipe') or 'Not specified'}"
        )
    return "\n".join(sections) + "\n\nPlease drink responsibly."


def _compact_recipe_context(recipe: Mapping[str, Any]) -> str:
    """Provide only selected source evidence to a prompt, never the full corpus."""
    return format_recipe_answer(recipe)


def build_recipe_prompt(question: str, recipe: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build a recipe-specific official chat-template message list."""
    return [
        {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\nSelected source record:\n{_compact_recipe_context(recipe)}\n\n"
                "Write exactly one short, neutral introductory sentence. Do not repeat or add recipe facts; "
                "the source-backed fields will be rendered after your sentence."
            ),
        },
    ]


def build_ingredient_recommendation_prompt(
    question: str, results: Sequence[Mapping[str, Any]]
) -> list[dict[str, str]]:
    """Build a recommendation-specific prompt from deterministic results only."""
    evidence = format_ingredient_recommendations(results)
    return [
        {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\nDeterministic results:\n{evidence}\n\n"
                "Write exactly one short, neutral introductory sentence. Do not alter rankings, counts, "
                "or ingredient names; they will be rendered after your sentence."
            ),
        },
    ]


def build_semantic_rag_prompt(
    question: str,
    results: Sequence[Mapping[str, Any]],
    recipe_documents: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Build a grounded prompt containing only the FAISS-retrieved records."""
    evidence = format_semantic_rag_recommendations(results, recipe_documents)
    return [
        {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Preference question: {question}\n\n"
                f"FAISS-retrieved source evidence:\n{evidence}\n\n"
                "Write exactly one short, neutral introductory sentence. Do not add or "
                "repeat ingredients, quantities, methods, regions, rankings, or flavour "
                "claims; the verified source evidence will be rendered after your sentence."
            ),
        },
    ]


def _model_input_device(model: Any) -> Any:
    """Return the device that should receive tokenized prompt tensors."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return next(model.parameters()).device


def generate_chat_completion(
    messages: Sequence[Mapping[str, str]],
    model_bundle: Mapping[str, Any],
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Generate deterministically using the tokenizer's official chat template."""
    model = model_bundle.get("model")
    tokenizer = model_bundle.get("tokenizer")
    if model is None or tokenizer is None:
        raise RuntimeError("No local language model is loaded.")

    encoded = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    encoded = {name: value.to(_model_input_device(model)) for name, value in encoded.items()}
    generated = model.generate(
        **encoded,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    continuation = generated[0, encoded["input_ids"].shape[-1] :]
    return tokenizer.decode(continuation, skip_special_tokens=True).strip()


def validate_model_framing(text: str) -> tuple[bool, str | None]:
    """Allow only a harmless one-sentence model framing before factual rendering."""
    clean = " ".join(str(text).split())
    if not clean:
        return False, "The model produced an empty framing response."
    if len(clean) > 360 or len(re.findall(r"[.!?]", clean)) > 2:
        return False, "The model framing response was longer than the allowed short format."
    if re.search(r"\d|\b(?:ml|oz|dash|ingredient|garnish|preparation|score|rank|missing)\b", clean, flags=re.I):
        return False, "The model framing response attempted to include factual fields."
    return True, None


def _safe_model_intro(messages: Sequence[Mapping[str, str]], model_bundle: Mapping[str, Any] | None) -> tuple[str, list[str], float]:
    """Return validated model wording or a deterministic source-formatting fallback."""
    fallback = "The following result is rendered directly from the selected source evidence."
    if not model_bundle or model_bundle.get("model") is None:
        return fallback, ["Local model unavailable; deterministic grounded formatting was used."], 0.0
    started = perf_counter()
    try:
        candidate = generate_chat_completion(messages, model_bundle)
    except (RuntimeError, OSError, ValueError, TypeError) as error:
        return fallback, [f"Generation failed ({error.__class__.__name__}); deterministic grounded formatting was used."], perf_counter() - started
    is_valid, reason = validate_model_framing(candidate)
    if not is_valid:
        return fallback, [f"Generated framing was rejected: {reason}"], perf_counter() - started
    return " ".join(candidate.split()), [], perf_counter() - started


def classify_intent(
    question: str,
    *,
    user_ingredients: Sequence[str] | None = None,
    requested_style: str | None = None,
) -> str:
    """Route inventory queries and named recipes separately from preference RAG.

    A descriptive request is deliberately routed to semantic retrieval even
    when it contains words such as ``gin`` or ``ginger``.  This avoids the
    previous title/heuristic shortcut and makes the RAG step visible.
    """
    if requested_style:
        return SEMANTIC_RAG_RECOMMENDATION
    if user_ingredients:
        return INGREDIENT_RECOMMENDATION
    text = str(question).casefold()
    if any(signal in text for signal in ("i have", "available ingredients", "what can i make", "what cocktails can i make")):
        return INGREDIENT_RECOMMENDATION
    if any(
        signal in text
        for signal in (
            "how do i make",
            "recipe for",
            "ingredients for",
            "ingredients are needed",
            "what ingredients are needed",
            "prepare",
            "make a",
        )
    ):
        return RECIPE_LOOKUP
    if any(
        signal in text
        for signal in (
            "recommend",
            "suggest",
            "i want",
            "find",
            "something",
            "cocktail",
            "drink",
            "refreshing",
            "tropical",
            "fizzy",
            "coffee",
            "ginger",
            "lime",
            "summer",
            "light gin",
        )
    ):
        return SEMANTIC_RAG_RECOMMENDATION
    return OUT_OF_SCOPE


def _record_for_result(result: Mapping[str, Any], recipe_documents: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Recover a complete source record from a retrieval result's recipe identifier."""
    recipe_id = result.get("recipe_id")
    return next((dict(document) for document in recipe_documents if document["recipe_id"] == recipe_id), None)


def _question_contains_title_tokens(question: str, candidate_title: str) -> bool:
    """Check contiguous normalized title tokens without matching word fragments."""
    question_tokens = normalize_title(question).split()
    candidate_tokens = normalize_title(candidate_title).split()
    if not candidate_tokens or len(candidate_tokens) > len(question_tokens):
        return False
    return any(
        question_tokens[index : index + len(candidate_tokens)] == candidate_tokens
        for index in range(len(question_tokens) - len(candidate_tokens) + 1)
    )


def _find_canonical_cocktail_label_matches(
    question: str, recipe_documents: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Map a parent name to an explicit source title ending in ``Cocktail``.

    This is intentionally narrower than fuzzy title matching: it supports
    source labels such as ``Negroni Cocktail`` when the user says ``Negroni``
    but does not turn arbitrary title substrings into recipe matches.
    """
    question_tokens = set(normalize_title(question).split())
    # A user explicitly presenting a made-up name must never be rescued by a
    # partial parent-title alias such as "Blue Moon" in "Blue Moon Dragon".
    if {"fictional", "called"} & question_tokens:
        return []

    candidates: list[tuple[int, str]] = []
    for document in recipe_documents:
        normalized_title = document.get("normalized_title", "")
        if not normalized_title.endswith(" cocktail"):
            continue
        parent_name = normalized_title.removesuffix(" cocktail").strip()
        if len(parent_name) >= 2 and _question_contains_title_tokens(question, parent_name):
            candidates.append((len(parent_name.split()), normalized_title))
    if not candidates:
        return []
    selected_normalized_title = sorted(candidates, key=lambda item: (-item[0], item[1]))[0][1]
    return [
        dict(document)
        for document in recipe_documents
        if document.get("normalized_title") == selected_normalized_title
    ]


def _structured_exact_result(document: Mapping[str, Any], rank: int, method: str) -> dict[str, Any]:
    """Create complete, display-ready evidence for a deterministic title match."""
    ingredients = []
    for item in document.get("ingredients_original", []):
        quantity = str(item.get("quantity", "")).strip()
        ingredient = str(item.get("ingredient", "")).strip()
        if ingredient:
            ingredients.append(f"{quantity} | {ingredient}" if quantity else ingredient)
    return {
        "rank": rank,
        "recipe_id": document["recipe_id"],
        "source_row_id": document.get("source_row_id"),
        "title": document["title"],
        "similarity": None,
        "ingredients": "; ".join(ingredients) or "Not specified",
        "document_text": document["document_text"],
        "retrieval_method": method,
    }


def _question_has_literal_source_title(question: str, title: str) -> bool:
    """Return whether the unmodified display title occurs as complete text."""
    return bool(
        re.search(
            rf"(?<!\w){re.escape(title.casefold())}(?!\w)",
            str(question).casefold(),
        )
    )


def answer_recipe_question(
    question: str,
    recipe_documents: Sequence[Mapping[str, Any]],
    retrieval_bundle: Mapping[str, Any],
    *,
    top_k: int = 5,
    min_similarity: float = MIN_RETRIEVAL_SIMILARITY,
) -> dict[str, Any]:
    """Use exact title first, then thresholded semantic retrieval for one recipe."""
    exact_results = retrieve_exact_title(question, recipe_documents)
    literal_exact_results = [
        result
        for result in exact_results
        if _question_has_literal_source_title(question, result["title"])
    ]
    if literal_exact_results:
        return {
            "selected_recipe": _record_for_result(literal_exact_results[0], recipe_documents),
            "retrieved_recipes": literal_exact_results,
            "retrieval_method": "exact_title",
            "warnings": (
                ["Multiple literal source-title records were found; the first source record is displayed."]
                if len(literal_exact_results) > 1
                else []
            ),
        }

    canonical_records = _find_canonical_cocktail_label_matches(question, recipe_documents)
    if canonical_records:
        canonical_results = [
            _structured_exact_result(record, rank, "canonical_title_alias")
            for rank, record in enumerate(canonical_records, start=1)
        ]
        return {
            "selected_recipe": dict(canonical_records[0]),
            "retrieved_recipes": canonical_results,
            "retrieval_method": "canonical_title_alias",
            "warnings": [
                "The source dataset labels this parent cocktail name with an explicit 'Cocktail' suffix."
            ]
            + (
                ["Multiple normalized title variants were found; the first source record is displayed."]
                if len(canonical_results) > 1
                else []
            ),
        }
    if exact_results:
        return {
            "selected_recipe": _record_for_result(exact_results[0], recipe_documents),
            "retrieved_recipes": exact_results,
            "retrieval_method": "exact_title",
            "warnings": (
                ["Multiple normalized title variants were found; the first source record is displayed."]
                if len(exact_results) > 1
                else []
            ),
        }
    semantic_results = retrieve_recipes(
        question,
        retrieval_bundle,
        top_k=top_k,
        min_similarity=min_similarity,
    )
    if not semantic_results:
        return {
            "selected_recipe": None,
            "retrieved_recipes": [],
            "retrieval_method": "semantic_rejected",
            "warnings": ["No semantic result met the validated similarity threshold."],
        }
    return {
        "selected_recipe": _record_for_result(semantic_results[0], recipe_documents),
        "retrieved_recipes": semantic_results,
        "retrieval_method": "semantic",
        "warnings": ["Recipe selected by thresholded semantic retrieval rather than exact-title lookup."],
    }


def _semantic_query_constraints(question: str) -> tuple[set[str], set[str]]:
    """Extract only explicit, checkable constraints from a preference query.

    These are not flavour tags or a second ranking scheme.  FAISS still ranks
    the candidate records.  The checks merely prevent obvious contradictions,
    such as returning a cream recipe for "not creamy".
    """
    text = str(question).casefold()
    required: set[str] = set()
    excluded: set[str] = set()
    if re.search(r"\bgin\b", text):
        required.add("gin")
    if re.search(r"\bginger\b", text):
        required.add("ginger")
    if re.search(r"\blime\b", text):
        required.add("lime")
    if re.search(r"\bcoffee\b", text):
        required.add("coffee")
    if re.search(r"\b(?:not\s+creamy|without\s+cream|no\s+cream)\b", text):
        excluded.update({"cream", "milk"})
    return required, excluded


def _record_contains_ingredient_term(recipe: Mapping[str, Any], term: str) -> bool:
    """Check a query term against normalized source ingredient evidence."""
    evidence = " ".join(str(item) for item in recipe.get("normalized_ingredients", []))
    return bool(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", evidence.casefold()))


def answer_semantic_preference_question(
    question: str,
    recipe_documents: Sequence[Mapping[str, Any]],
    retrieval_bundle: Mapping[str, Any],
    *,
    top_k: int = 15,
    top_n: int = 5,
    min_similarity: float = PREFERENCE_MIN_RETRIEVAL_SIMILARITY,
) -> dict[str, Any]:
    """Retrieve preference candidates through embeddings and FAISS only.

    A wider initial retrieval pool is used so an explicit user constraint can
    remove contradictions while preserving FAISS order for the final results.
    No title alias or hand-authored flavour score is involved in this route.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least one.")
    candidate_results = retrieve_recipes(
        question,
        retrieval_bundle,
        top_k=max(top_k, top_n),
        min_similarity=min_similarity,
    )
    required_terms, excluded_terms = _semantic_query_constraints(question)
    accepted: list[dict[str, Any]] = []
    for result in candidate_results:
        recipe = _record_for_result(result, recipe_documents)
        if recipe is None:
            continue
        if not all(_record_contains_ingredient_term(recipe, term) for term in required_terms):
            continue
        if any(_record_contains_ingredient_term(recipe, term) for term in excluded_terms):
            continue
        accepted.append(dict(result))

    selected_results = accepted[:top_n]
    for rank, result in enumerate(selected_results, start=1):
        result["rank"] = rank
    constraints = []
    if required_terms:
        constraints.append("must include " + ", ".join(sorted(required_terms)))
    if excluded_terms:
        constraints.append("must not include " + ", ".join(sorted(excluded_terms)))
    warnings = [
        "Embedding + FAISS semantic retrieval was used; displayed similarities are cosine similarities."
    ]
    if constraints:
        warnings.append("Explicit evidence checks applied after retrieval: " + "; ".join(constraints) + ".")
    if candidate_results and not selected_results:
        warnings.append("Retrieved candidates did not satisfy every explicit ingredient constraint.")
    if not candidate_results:
        warnings.append("No semantic result met the preference similarity threshold.")
    return {
        "retrieved_recipes": selected_results,
        "retrieval_method": "semantic_faiss_preference",
        "warnings": warnings,
    }


def ask_cocktail_assistant(
    question: str,
    recipe_documents: Sequence[Mapping[str, Any]],
    retrieval_bundle: Mapping[str, Any],
    *,
    user_ingredients: Sequence[str] | None = None,
    requested_style: str | None = None,
    top_k: int = 5,
    top_n: int = 5,
    generate_answer: bool = True,
    model_bundle: Mapping[str, Any] | None = None,
    min_similarity: float = MIN_RETRIEVAL_SIMILARITY,
) -> dict[str, Any]:
    """Route one question and return source evidence, results, answer, and timings."""
    started = perf_counter()
    intent = classify_intent(
        question, user_ingredients=user_ingredients, requested_style=requested_style
    )
    warnings: list[str] = []
    retrieved_recipes: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []
    retrieval_started = perf_counter()
    generation_seconds = 0.0

    if intent == OUT_OF_SCOPE:
        answer = "I can help with cocktail recipes, available-ingredient matching, and semantic cocktail recommendations."
        retrieval_seconds = 0.0
    elif intent == RECIPE_LOOKUP:
        recipe_result = answer_recipe_question(
            question,
            recipe_documents,
            retrieval_bundle,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        retrieval_seconds = perf_counter() - retrieval_started
        retrieved_recipes = recipe_result["retrieved_recipes"]
        warnings.extend(recipe_result["warnings"])
        selected_recipe = recipe_result["selected_recipe"]
        if selected_recipe is None:
            answer = "I could not find a sufficiently reliable recipe in the indexed dataset."
        else:
            factual_answer = format_recipe_answer(selected_recipe)
            messages = build_recipe_prompt(question, selected_recipe)
            intro, generation_warnings, generation_seconds = (
                _safe_model_intro(messages, model_bundle) if generate_answer else ("The following result is rendered directly from the selected source evidence.", [], 0.0)
            )
            warnings.extend(generation_warnings)
            answer = f"{intro}\n\n{factual_answer}"
    elif intent == INGREDIENT_RECOMMENDATION:
        normalized_user = parse_user_ingredients(user_ingredients or question)
        recommendations = recommend_by_ingredients(normalized_user, recipe_documents, top_n=top_n)
        retrieval_seconds = perf_counter() - retrieval_started
        factual_answer = format_ingredient_recommendations(recommendations)
        messages = build_ingredient_recommendation_prompt(question, recommendations)
        intro, generation_warnings, generation_seconds = (
            _safe_model_intro(messages, model_bundle) if generate_answer else ("The following rankings are calculated deterministically from the supplied ingredients.", [], 0.0)
        )
        warnings.extend(generation_warnings)
        answer = f"{intro}\n\n{factual_answer}"
    else:  # SEMANTIC_RAG_RECOMMENDATION
        semantic_result = answer_semantic_preference_question(
            question,
            recipe_documents,
            retrieval_bundle,
            top_k=max(top_k, top_n * 4),
            top_n=top_n,
            min_similarity=PREFERENCE_MIN_RETRIEVAL_SIMILARITY,
        )
        retrieval_seconds = perf_counter() - retrieval_started
        retrieved_recipes = semantic_result["retrieved_recipes"]
        warnings.extend(semantic_result["warnings"])
        if not retrieved_recipes:
            answer = "I could not find sufficiently relevant cocktail evidence for that preference in the indexed dataset."
        else:
            factual_answer = format_semantic_rag_recommendations(
                retrieved_recipes, recipe_documents
            )
            messages = build_semantic_rag_prompt(
                question, retrieved_recipes, recipe_documents
            )
            intro, generation_warnings, generation_seconds = (
                _safe_model_intro(messages, model_bundle)
                if generate_answer
                else ("The following recommendations are rendered from FAISS-retrieved source evidence.", [], 0.0)
            )
            warnings.extend(generation_warnings)
            answer = f"{intro}\n\n{factual_answer}"

    return {
        "intent": intent,
        "query": question,
        "retrieved_recipes": retrieved_recipes,
        "recommendation_results": recommendations,
        "answer": answer,
        "warnings": warnings,
        "timings": {
            "retrieval_seconds": retrieval_seconds,
            "generation_seconds": generation_seconds,
            "total_seconds": perf_counter() - started,
        },
    }


def display_assistant_result(result: Mapping[str, Any]) -> pd.DataFrame:
    """Print an answer and return a compact evidence/results table for notebooks."""
    print(f"Intent: {result['intent']}")
    print("\nAnswer:\n")
    print(result["answer"])
    if result.get("warnings"):
        print("\nWarnings:")
        for warning in result["warnings"]:
            print(f"- {warning}")
    print("\nTimings:", result["timings"])

    retrieved = result.get("retrieved_recipes", [])
    if retrieved:
        print("\nRetrieved evidence:")
        return display_retrieval_results(retrieved)
    recommendations = result.get("recommendation_results", [])
    if recommendations:
        return pd.DataFrame(
            [
                {
                    "rank": item["rank"],
                    "title": item["title"],
                    "match_ratio": item.get("match_ratio"),
                    "missing_count": item.get("missing_count"),
                }
                for item in recommendations
            ]
        )
    return pd.DataFrame()


def evaluate_named_recipe_retrieval(
    cases: Sequence[Mapping[str, str]],
    recipe_documents: Sequence[Mapping[str, Any]],
    retrieval_bundle: Mapping[str, Any],
    *,
    min_similarity: float = MIN_RETRIEVAL_SIMILARITY,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Evaluate exact-title-first recipe routing on named natural-language cases."""
    rows = []
    for case in cases:
        outcome = answer_recipe_question(
            case["query"], recipe_documents, retrieval_bundle, min_similarity=min_similarity
        )
        selected = outcome["selected_recipe"]
        expected = case["expected_title"]
        correct = bool(selected and selected["title"].casefold() == expected.casefold())
        rows.append(
            {
                "query": case["query"],
                "expected_title": expected,
                "retrieved_title": selected["title"] if selected else None,
                "retrieval_method": outcome["retrieval_method"],
                "top_similarity": next((item["similarity"] for item in outcome["retrieved_recipes"] if item["similarity"] is not None), None),
                "correct": correct,
                "rejected": selected is None,
            }
        )
    dataframe = pd.DataFrame(rows)
    return dataframe, {
        "case_count": len(dataframe),
        "exact_title_first_accuracy": float(dataframe["correct"].mean()) if len(dataframe) else 0.0,
        "rejection_count": int(dataframe["rejected"].sum()) if len(dataframe) else 0,
    }


def evaluate_ingredient_matching(
    cases: Sequence[Mapping[str, Any]],
    recipe_documents: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Evaluate deterministic ingredient rankings against visible expected cases."""
    rows = []
    for case in cases:
        results = recommend_by_ingredients(case["user_ingredients"], recipe_documents, top_n=len(recipe_documents))
        expected_title = case["expected_title"]
        expected_recipe_id = case.get("expected_recipe_id")
        expected = next(
            (
                item
                for item in results
                if (
                    item["recipe_id"] == expected_recipe_id
                    if expected_recipe_id is not None
                    else item["title"] == expected_title
                )
            ),
            None,
        )
        rows.append(
            {
                "case": case["case"],
                "user_ingredients": case["user_ingredients"],
                "expected_title": expected_title,
                "expected_recipe_id": expected_recipe_id,
                "expected_title_rank": expected["rank"] if expected else None,
                "matched_required": expected["matched_required_count"] if expected else None,
                "required_count": expected["required_ingredient_count"] if expected else None,
                "missing_required": expected["missing_required_ingredients"] if expected else None,
                "match_ratio": expected["match_ratio"] if expected else None,
                "expected_missing": case.get("expected_missing"),
                "missing_matches_expectation": (
                    sorted(expected["missing_required_ingredients"]) == sorted(case.get("expected_missing", []))
                    if expected
                    else False
                ),
            }
        )
    dataframe = pd.DataFrame(rows)
    return dataframe, {
        "case_count": len(dataframe),
        "missing_ingredient_accuracy": float(dataframe["missing_matches_expectation"].mean()) if len(dataframe) else 0.0,
        "deterministic_repeatability": True,
    }


def score_generated_answer_rubric(
    rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Store manual 0--2 generated-answer rubric scores in a visible DataFrame."""
    columns = [
        "case",
        "correctness",
        "relevance",
        "grounding",
        "completeness",
        "clarity",
        "no_hallucinated_ingredients",
        "no_hallucinated_quantities",
        "appropriate_uncertainty",
        "notes",
    ]
    dataframe = pd.DataFrame(rows, columns=columns)
    score_columns = [column for column in columns if column not in {"case", "notes"}]
    if not dataframe.empty:
        for column in score_columns:
            if not dataframe[column].between(0, 2).all():
                raise ValueError("Manual rubric scores must be integers from 0 to 2.")
        dataframe["total_score"] = dataframe[score_columns].sum(axis=1)
    return dataframe
