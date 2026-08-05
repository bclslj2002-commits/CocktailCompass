# CocktailCompass: grounded cocktail RAG assistant

CocktailCompass is an English-only CP5403 Assessment 2 project. It answers
cocktail questions from the public `erwanlc/cocktails_recipe` dataset without
an API key, paid inference service, vector database, or agent framework.

## Supported use cases

1. Complete recipe lookup: exact-title matching is preferred, then thresholded
   semantic retrieval returns one source record with its original quantities,
   glass, garnish, and preparation method.
2. Ingredient recommendations: deterministic set matching ranks recipes,
   shows match ratios, and lists every missing required ingredient.
3. Semantic preference recommendations: descriptive questions are encoded with
   the same embedding model as the recipe documents, searched in FAISS, and
   answered only from the retrieved source records. Explicit constraints such
   as `gin`, `ginger`, `lime`, `coffee`, and `not creamy` are checked against
   source ingredient evidence after retrieval; no flavour-tag column is used.

Fictional cocktail names with insufficient evidence and unrelated questions are
rejected rather than answered from general cocktail knowledge.

## Architecture

```text
Dataset row -> safe parsing and normalization -> complete recipe document
                                                  |
                                all-MiniLM-L6-v2 embeddings (L2 normalized)
                                                  |
                                         FAISS IndexFlatIP
                                                  |
 exact title / canonical "... Cocktail" label / deterministic ingredient matching
                                                  |
      descriptive preference -> embedding query -> FAISS semantic retrieval
                                                  |
                   selected source evidence -> grounded Qwen framing
                                                  |
                         source-rendered factual response and visible evidence
```

For a preference query, Qwen receives only the FAISS-retrieved recipe records.
It never invents recipe facts, fills missing ingredients, changes quantities,
or changes retrieval order. It produces only a short grounded framing sentence;
the source-backed content is rendered by Python and unsafe framing is replaced
with deterministic wording.

## Dataset and preparation

- Dataset: [`erwanlc/cocktails_recipe`](https://huggingface.co/datasets/erwanlc/cocktails_recipe)
- Accessed: 2026-07-23
- Dataset card: English cocktail recipes scraped from Difford's Cocktail Guide;
  the card reports licence `other`, so it must be verified before reuse beyond
  this assessment.
- Actual cached split: 6,956 rows with string fields `title`, `glass`,
  `garnish`, `recipe`, and `ingredients`.
- Retained documents: 6,939 after removing one missing-core record and 16 exact
  duplicate source records. Duplicate titles and variants remain traceable.

Ingredient strings are parsed only with `ast.literal_eval`. Normalization is
conservative: source display data and brands are preserved. The aliases are
explicitly limited to `light rum -> white rum`, `simple syrup -> sugar syrup`,
and `club soda -> soda water`.

## Models and hardware

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
- Index: `faiss.IndexFlatIP` over L2-normalized vectors, so inner product is
  cosine similarity.
- Final local model: [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), 4B parameters, Apache-2.0 according to its model card.
- CUDA loading: 4-bit bitsandbytes NF4, double quantization, float16 compute,
  and `device_map="auto"`.
- Documented fallback: `microsoft/Phi-4-mini-instruct`, attempted only if Qwen
  fails to load; the change and exception are displayed.

The executed evidence used one NVIDIA GeForce RTX 4060 Laptop GPU (8,187 MiB),
PyTorch `2.7.1+cu126`, Transformers `5.14.1`, and bitsandbytes `0.49.2`.
CPU-only systems retain deterministic responses but should use a Kaggle or
Colab GPU runtime for final model evaluation.

## Run the notebook

1. Create a Python 3.12 environment, or open a Kaggle/Colab GPU notebook.
2. Install `requirements.txt`. For a local CUDA system, use the appropriate
   vendor PyTorch wheel; the tested Windows configuration is CUDA 12.6.
3. Run `notebooks/AI_Home_Mixologist_qwen_grounded_generation.ipynb` from the first cell to the last.
4. Review the retrieved source evidence, evaluation DataFrames, and the final
   reproducibility assertions.

The first run downloads public dataset and model weights to `work/hf_cache`.
This ignored directory contains no credentials and must not be committed.

## Archived teaching baselines

`AI_Home_Mixologist.ipynb` and `notebooks/cocktailcompass_colab.ipynb` are
preserved, non-final teaching/reference notebooks. They contain legacy
FLAN-T5-style baseline code and are **not** part of the final implementation,
evaluation, requirements, or submission run path. The only final notebook is
`notebooks/AI_Home_Mixologist_qwen_grounded_generation.ipynb`, which uses Qwen as documented above.

Example questions:

```python
ask_cocktail_assistant("How do I make a Mojito?", recipe_documents, retrieval_bundle)
ask_cocktail_assistant(
    "What cocktails can I make?",
    recipe_documents,
    retrieval_bundle,
    user_ingredients=["white rum", "lime juice", "mint", "simple syrup", "soda water"],
)
ask_cocktail_assistant(
    "Recommend a refreshing cocktail with citrus and herbs.",
    recipe_documents,
    retrieval_bundle,
)
ask_cocktail_assistant("I want something tropical and fizzy.", recipe_documents, retrieval_bundle)
ask_cocktail_assistant("Suggest a coffee-flavoured cocktail that is not creamy.", recipe_documents, retrieval_bundle)
ask_cocktail_assistant("Find a light gin cocktail for summer.", recipe_documents, retrieval_bundle)
ask_cocktail_assistant("Recommend a cocktail with ginger and lime.", recipe_documents, retrieval_bundle)
```

## Measured evaluation summary

The executed notebook displays all rows in pandas DataFrames. On the cached
RTX 4060 run:

- Named recipe routing: 5/5 correct after exact title plus the explicit
  `... Cocktail` parent-label rule.
- Ingredient matching: 5/5 expected missing-ingredient sets correct;
  repeatability check passed.
- Semantic preference examples display the FAISS cosine similarity, full
  retrieved evidence, explicit constraint checks, and the resulting source
  records. Re-run the final notebook to regenerate the evaluation table after
  any model or data-cache change.
- Grounding rubric: four inspected responses scored 16/16 because factual
  fields are deterministic source renderings. Qwen framing was rejected in
  three demonstrations when it attempted factual wording; the deterministic
  fallback preserved the answer safely.
- Efficiency: 6,939 recipes indexed, 384-dimensional embeddings, 4.76 s for
  cached embedding/index construction, 8.92 s model loading, 0.13 s mean
  deterministic retrieval, 1.33 s mean generation, 1.46 s mean end-to-end
  response time, and 2,893.7 MiB peak allocated GPU memory across the eight
  demonstrations.

The notebook also retains the earlier semantic-only baseline: strict top-1
accuracy 0.0, strict top-3 0.2, and MRR 0.1 over five intentionally difficult
natural-language cases. These results support exact-title priority and careful
threshold calibration rather than a claim that semantic retrieval is reliable
by itself.

## Limitations and responsible use

- The source data can contain noise, duplicates, missing values, inconsistent
  ingredient wording, brand-specific ingredients, and ambiguous garnishes.
- Brand names are deliberately not equated with generic spirits; generic
  ingredient lists can therefore miss branded source recipes.
- Matching does not check the quantities a user has. Optional ingredients and
  pantry defaults are limited, explicit design choices.
- Semantic preference retrieval can return weak matches for subjective wording
  such as "refreshing" or "tropical". Similarity is evidence of semantic
  closeness, not a verified flavour category or cultural claim.
- Retrieval thresholds and parent-title aliases require further validation on a
  larger held-out set. The model may still produce unsupported wording, though
  validation and deterministic formatting reduce this risk.
- Check recipes before real-world preparation. The project provides no medical
  or health advice; alcohol has no claimed health benefit, and responsible use
  remains the user's responsibility.

## Licence and acknowledgement

The Qwen model card reports Apache-2.0. The dataset card reports `other`, so
the original source and licence conditions require separate verification. This
repository stores no downloaded model files, Hugging Face tokens, API keys, or
other credentials.

Implementation and documentation were developed with assistance from OpenAI
Codex. The student should review, run, understand, and be able to explain every
implementation decision and recorded result before submission.
