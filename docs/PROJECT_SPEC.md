# CP5403 Assessment 2 Project Specification

Build a complete, clear, reproducible RAG-based cocktail assistant for CP5403 Assessment 2.

Before implementing the system, read this complete specification carefully and treat it as the source of truth for the project.

Save this specification in:

```text
docs/PROJECT_SPEC.md
```

Do not overwrite the reference notebook.

For the first task, complete only the repository inspection and implementation plan described in the final section. Do not implement the whole project in one pass.

## Current implementation override: semantic preference RAG

This override supersedes the earlier refreshing-heuristic requirements in this
historical planning document. The final public project keeps deterministic
exact-title lookup and available-ingredient matching, but all descriptive
preference questions must use the RAG pathway:

```text
natural-language preference -> embedding -> FAISS retrieval -> retrieved recipe evidence -> Qwen framing
```

The following questions are mandatory visible demonstrations of this route:

- `Recommend a refreshing cocktail with citrus and herbs.`
- `I want something tropical and fizzy.`
- `Suggest a coffee-flavoured cocktail that is not creamy.`
- `Find a light gin cocktail for summer.`
- `Recommend a cocktail with ginger and lime.`

The assistant retrieves a larger semantic candidate pool, preserves FAISS
order, and displays the selected records and cosine similarities. It may apply
only explicit, source-verifiable constraints such as `gin`, `ginger`, `lime`,
`coffee`, and `not creamy`; it must not use manually curated flavour-tag
columns or a hand-authored refreshing-score ranking. Qwen receives only the
retrieved evidence and must not add facts. The old refreshing heuristic may be
retained solely as a historical offline-analysis helper, not as a user-facing
recommendation pathway.

---

# 1. Project title

**Cocktail Recipe and Recommendation RAG Assistant**

The project is an English-only hybrid RAG application based on a cocktail recipe dataset.

All user questions, system prompts, generated answers, code comments, notebook Markdown, demonstrations, and evaluation questions must be written in English.

Chinese-language input and multilingual retrieval are outside the project scope.

---

# 2. Reference implementation

Use the following repository and notebook as a conceptual reference:

```text
Repository:
Georgegiri/cp5403-rag-assessment-helper

Reference notebook:
notebooks/simple_rag_assisstant_clean.ipynb
```

Do not copy the reference notebook directly.

Preserve its main educational ideas:

- use a SentenceTransformer embedding model;
- use FAISS as the vector index;
- use an open-weights Hugging Face language model;
- retrieve relevant records before generation;
- display retrieved evidence;
- constrain generation to retrieved information;
- handle insufficient evidence safely;
- keep the implementation understandable for a student presentation.

The new project must extend the reference design substantially by adding structured ingredient matching, exact-title lookup, explainable style classification, systematic evaluation, and a stronger language model.

Do not use:

- OpenAI API;
- paid inference APIs;
- paid vector databases;
- proprietary hosted language models;
- API keys for generation;
- unnecessary agent frameworks;
- LangChain unless there is a clearly documented reason that cannot be handled with direct Python functions.

Prefer direct Python, Transformers, SentenceTransformers, FAISS, pandas, NumPy, and Hugging Face Datasets.

---

# 3. Dataset

Use the Hugging Face dataset:

```text
erwanlc/cocktails_recipe
```

Load it with:

```python
from datasets import load_dataset

dataset = load_dataset("erwanlc/cocktails_recipe")
```

Expected fields may include:

- `title`
- `glass`
- `garnish`
- `recipe`
- `ingredients`

Do not assume the schema or data types before inspecting the loaded dataset.

The notebook must display:

- dataset object;
- split names;
- row count;
- column names;
- one raw example;
- data types;
- missing-value counts;
- duplicate-title counts;
- duplicate-record counts.

If `ingredients` is stored as a string representation of a nested Python structure, parse it safely with:

```python
ast.literal_eval()
```

Never use `eval()`.

Handle:

- null fields;
- empty strings;
- malformed ingredient entries;
- duplicate recipes;
- duplicate cocktail names;
- inconsistent capitalisation;
- punctuation;
- bracketed descriptions;
- brand names;
- optional ingredients;
- singular and plural wording;
- measurements mixed with ingredient names.

Keep both:

1. the original ingredient data for display;
2. normalized ingredient names for matching.

Do not silently delete a large amount of data. Report how many records were retained, removed, or repaired at each cleaning step.

Document the dataset source, dataset card, licence status, access date, preparation steps, and limitations in the notebook and README.

---

# 4. System architecture

The project is a hybrid system with five components:

1. Exact cocktail-title matching
2. Semantic recipe retrieval
3. Deterministic ingredient matching
4. Explainable refreshing-style scoring
5. Grounded language-model generation

The language model must not make the core recommendation decisions.

Use deterministic Python logic for:

- exact-title matching;
- ingredient overlap;
- missing-ingredient calculation;
- recommendation ranking;
- refreshing-score calculation;
- out-of-scope detection where practical.

Use the language model only to:

- convert retrieved information into a natural answer;
- explain deterministic recommendation results;
- present retrieved facts clearly.

The source recipe fields and recommendation scores must remain the source of truth.

---

# 5. Recipe document design

Treat each cocktail recipe as one complete document.

Do not divide one short recipe into separate ingredient, garnish, and instruction chunks. These fields must remain together.

Construct document text similar to:

```text
Title: Mojito
Glass: Collins glass
Garnish: Mint sprig

Ingredients:
- 50 ml | White rum
- 25 ml | Lime juice
- 15 ml | Sugar syrup
- Mint leaves
- Soda water

Preparation:
Complete preparation instructions from the source record.
```

Store metadata similar to:

```python
{
    "recipe_id": ...,
    "title": ...,
    "normalized_title": ...,
    "glass": ...,
    "garnish": ...,
    "recipe": ...,
    "ingredients_original": ...,
    "required_ingredients": ...,
    "optional_ingredients": ...,
    "normalized_ingredients": ...,
    "document_text": ...
}
```

Each source record must remain traceable to the original dataset row.

---

# 6. Embedding model and FAISS

Use this embedding model as the initial retrieval baseline:

```python
sentence-transformers/all-MiniLM-L6-v2
```

Because the application is English-only, multilingual embeddings are unnecessary.

Generate embeddings from complete recipe documents.

Normalize document and query embeddings before indexing.

Use cosine similarity through:

```python
faiss.IndexFlatIP
```

Apply L2 normalization before adding vectors and before querying:

```python
faiss.normalize_L2(embeddings)
faiss.normalize_L2(query_embedding)
```

The semantic retrieval function must return structured results rather than only printing them.

Implement functions similar to:

```python
build_faiss_index(recipe_documents)
retrieve_recipes(query, top_k=5, min_similarity=None)
display_retrieval_results(results)
```

Each retrieval result should include:

```python
{
    "rank": ...,
    "recipe_id": ...,
    "title": ...,
    "similarity": ...,
    "ingredients": ...,
    "document_text": ...
}
```

Display:

- retrieval rank;
- cocktail title;
- cosine similarity;
- ingredients;
- shortened document preview.

Do not show only the final language-model answer.

---

# 7. Exact-title lookup

Recipe questions should first use deterministic title matching.

Implement:

```python
normalize_title(title)
extract_known_cocktail_title(question, known_titles)
find_cocktail_by_title(cocktail_name)
```

Recommended procedure:

1. Normalize the user question.
2. Check whether any known cocktail title appears in the question.
3. Prefer the longest valid title match.
4. Perform case-insensitive exact matching.
5. If no reliable title match is found, use semantic retrieval.
6. Reject weak retrieval matches below a validated similarity threshold.

Do not use the language model to invent a cocktail name from an ambiguous question.

---

# 8. Core feature 1: complete recipe lookup

Example questions:

```text
How do I make a Mojito?
Give me the recipe for a Mojito.
What ingredients are needed for a Negroni?
How should I prepare an Espresso Martini?
```

The answer must contain:

- cocktail name;
- glass type;
- complete ingredients and quantities;
- preparation method;
- garnish;
- retrieved source title.

The answer must not add:

- unlisted ingredients;
- invented quantities;
- invented preparation steps;
- unsupported substitutions;
- unsupported serving advice.

Implement:

```python
answer_recipe_question(question, top_k=5)
```

Use exact-title matching first and semantic retrieval second.

Where possible, render ingredients and quantities deterministically from the source data. The language model may improve presentation, but it must not rewrite the factual fields incorrectly.

When no sufficiently reliable recipe is found, return:

```text
I could not find a sufficiently reliable recipe in the indexed dataset.
```

Do not generate a plausible recipe for a fictional cocktail.

---

# 9. Core feature 2: recommendations from available ingredients

Example questions:

```text
I have white rum, lime juice, mint, sugar syrup, and soda water. What can I make?
I have gin and lemon juice. What cocktails are closest?
What can I make with vodka, orange juice, and lime juice?
```

This feature must use deterministic set-based ingredient matching.

It must not rely only on semantic similarity or language-model judgement.

Implement:

```python
normalize_ingredient_name(name)
parse_user_ingredients(text)
get_recipe_ingredient_sets(recipe)
score_ingredient_match(recipe, user_ingredients)
recommend_by_ingredients(user_ingredients, top_n=5)
```

For every recipe calculate:

- required ingredients;
- optional ingredients;
- matched required ingredients;
- missing required ingredients;
- matched optional ingredients;
- required ingredient count;
- matched required ingredient count;
- missing count;
- match ratio.

Use:

```python
match_ratio = matched_required_count / required_ingredient_count
```

Protect against division by zero.

Use this ranking order:

1. recipes with zero missing required ingredients;
2. higher match ratio;
3. lower missing count;
4. lower required ingredient count;
5. alphabetical title as a stable final tie-breaker.

Optional ingredients must not cause a recipe to be classified as impossible.

Garnishes should not count as required unless they also appear as a required recipe ingredient.

Treat only clearly defined basic defaults, such as ice or plain water, as configurable pantry defaults. Do not treat soda water, tonic water, fruit juice, spirits, syrups, or liqueurs as automatically available.

Create an explicit, reviewable alias dictionary, for example:

```python
INGREDIENT_ALIASES = {
    "light rum": "white rum",
    "sugar syrup": "simple syrup",
    "club soda": "soda water",
}
```

Do not make broad or unsafe equivalences.

Examples of invalid equivalences:

- gin is not vodka;
- tequila is not rum;
- lemon juice is not automatically lime juice;
- tonic water is not soda water;
- cream is not milk;
- sparkling wine is not automatically champagne.

Any approximate equivalence must be documented and included in the limitations.

Expected output structure:

```text
Best matches based on your ingredients:

1. Cocktail name
   Match: 5/5 required ingredients
   Status: You can make this cocktail now
   Available ingredients: ...
   Missing required ingredients: None
   Optional ingredients not available: ...

2. Cocktail name
   Match: 3/4 required ingredients
   Status: One required ingredient is missing
   Missing required ingredients: ...
```

If no exact makeable recipe exists, return the closest recipes and their missing ingredients.

The language model may explain the results, but it must not recalculate or alter the deterministic ranking.

---

# 10. Core feature 3: refreshing cocktail recommendations

Example questions:

```text
Recommend a refreshing cocktail.
I want something light and refreshing.
Recommend a refreshing cocktail with gin.
What is a refreshing drink containing cucumber?
```

The dataset does not provide an official `refreshing` label.

Therefore, implement a transparent ingredient-based heuristic rather than presenting refreshing style as a dataset fact.

Create explicit weighted dictionaries:

```python
REFRESHING_POSITIVE_SIGNALS = {
    "lime juice": ...,
    "lemon juice": ...,
    "grapefruit juice": ...,
    "mint": ...,
    "cucumber": ...,
    "soda water": ...,
    "tonic water": ...,
    "sparkling water": ...,
    "sparkling wine": ...,
}
```

```python
REFRESHING_NEGATIVE_SIGNALS = {
    "cream": ...,
    "milk": ...,
    "coffee liqueur": ...,
    "chocolate": ...,
    "egg": ...,
}
```

The heuristic may also use documented preparation or serving signals such as:

- served over ice;
- topped with soda;
- highball glass;
- Collins glass;
- citrus-forward preparation.

Do not classify a drink as refreshing only because it contains alcohol or juice.

Implement:

```python
calculate_refreshing_score(recipe)
recommend_by_style(
    style="refreshing",
    required_ingredients=None,
    top_n=5
)
```

Return:

- cocktail title;
- refreshing score;
- positive signals found;
- negative signals found;
- score calculation;
- short explanation;
- complete source recipe when selected.

Every refreshing recommendation must display this disclosure:

```text
This recommendation uses an ingredient-based freshness heuristic because the source dataset does not contain an official refreshing category.
```

For a query such as:

```text
Recommend a refreshing cocktail with gin.
```

first filter recipes that contain gin, then rank the filtered recipes by refreshing score.

Do not allow the language model to choose drinks that were not returned by the deterministic ranking.

---

# 11. Language model

Use an open-weights Hugging Face causal language model.

Primary model:

```python
Qwen/Qwen3-4B-Instruct-2507
```

Compatibility fallback:

```python
microsoft/Phi-4-mini-instruct
```

Do not use `google/flan-t5-base` as the final project model. It may be retained only as a lightweight baseline comparison when useful.

The final selected model must:

- run locally in the notebook;
- require no paid API;
- require no OpenAI key;
- support English instruction following;
- fit the available Kaggle or Colab GPU through quantization;
- use its official chat template;
- be documented with model name, parameter count, model card, licence, hardware method, and limitations.

For Qwen, use:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
```

Use a sufficiently recent Transformers version compatible with the model.

On CUDA, attempt 4-bit loading with:

```python
from transformers import BitsAndBytesConfig
```

Use a configuration similar to:

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
```

Use:

```python
device_map="auto"
```

The notebook must print:

- selected model;
- quantization mode;
- CUDA availability;
- GPU name;
- number of available GPUs;
- model device map where available.

The implementation must not require two GPUs. It should be capable of running on one T4-class GPU with an appropriate quantized model configuration.

If 4-bit loading fails:

1. report the exact error;
2. attempt a documented compatibility fallback;
3. do not silently change the final model;
4. record the fallback in the notebook.

On CPU-only environments, provide a clear warning that final generation will be slow. A smaller model may be used for a smoke test, but final evaluation should use the declared final model on GPU.

Use deterministic generation by default:

```python
do_sample=False
```

Use a reasonable output limit such as:

```python
max_new_tokens=300
```

Do not use an unnecessarily large context window.

Pass only the retrieved or selected recipe records to the model.

Never pass the complete dataset into the generation prompt.

---

# 12. Grounded generation prompt

Create a system prompt that tells the model:

1. Use only the supplied cocktail records and deterministic recommendation results.
2. Do not rely on general cocktail knowledge.
3. Do not add ingredients, quantities, garnishes, or preparation steps.
4. Do not change ingredient-match calculations.
5. Do not change refreshing scores.
6. Clearly distinguish source facts from heuristic interpretation.
7. State when the retrieved evidence is insufficient.
8. Do not answer unrelated questions.
9. Do not claim alcohol has health benefits.
10. Include a brief responsible-drinking note where appropriate.
11. Include a safety warning when the retrieved preparation explicitly involves flame or another hazardous step.
12. Answer in English only.

Create separate prompt builders where helpful:

```python
build_recipe_prompt(...)
build_ingredient_recommendation_prompt(...)
build_style_recommendation_prompt(...)
```

Do not create one unstructured prompt containing every possible instruction.

Where practical, perform a grounding validation after generation:

- source titles in the answer must come from retrieved results;
- stated ingredient names should be supported by retrieved records;
- quantities should match source data;
- unsupported content should trigger a deterministic fallback answer.

---

# 13. Intent routing

Implement a simple, transparent intent router.

Use these intents:

```python
RECIPE_LOOKUP
INGREDIENT_RECOMMENDATION
STYLE_RECOMMENDATION
OUT_OF_SCOPE
```

Use explicit parameters first and keyword rules second.

Example English signals:

```text
RECIPE_LOOKUP:
"how do I make"
"recipe for"
"ingredients for"
"prepare"
"make a"

INGREDIENT_RECOMMENDATION:
"I have"
"available ingredients"
"what can I make"
"what cocktails can I make"

STYLE_RECOMMENDATION:
"refreshing"
"light and refreshing"
"fresh"
"refreshing cocktail"

OUT_OF_SCOPE:
questions unrelated to cocktails, recipes, ingredients, or cocktail styles
```

Do not train a separate intent-classification model.

Implement:

```python
classify_intent(
    question,
    user_ingredients=None,
    requested_style=None
)
```

Create one public interface:

```python
ask_cocktail_assistant(
    question,
    user_ingredients=None,
    requested_style=None,
    top_k=5,
    top_n=5,
    generate_answer=True
)
```

Return a structured dictionary:

```python
{
    "intent": ...,
    "query": ...,
    "retrieved_recipes": ...,
    "recommendation_results": ...,
    "answer": ...,
    "warnings": ...,
    "timings": {
        "retrieval_seconds": ...,
        "generation_seconds": ...,
        "total_seconds": ...
    }
}
```

Create a separate display function:

```python
display_assistant_result(result)
```

Core functions must return data and must not depend entirely on `print()`.

---

# 14. Notebook structure

Create:

```text
notebooks/cocktail_rag_assistant.ipynb
```

Use this order:

1. Project title and objective
2. Scope and research question
3. System architecture
4. Dependency installation
5. Imports and reproducibility
6. Runtime and GPU inspection
7. Load the Hugging Face dataset
8. Inspect raw schema
9. Data-quality analysis
10. Parse ingredients
11. Normalize ingredient names
12. Handle invalid and duplicate records
13. Build recipe documents
14. Load embedding model
15. Generate normalized embeddings
16. Build FAISS index
17. Implement exact-title matching
18. Implement semantic retrieval
19. Implement ingredient recommendation
20. Implement refreshing heuristic
21. Load the language model
22. Build grounded prompts
23. Implement intent routing
24. Implement unified assistant
25. Demonstration examples
26. Retrieval evaluation
27. Ingredient-matching evaluation
28. Refreshing-heuristic evaluation
29. Generation and grounding evaluation
30. Baseline comparison
31. Efficiency results
32. Failure cases
33. Limitations
34. Conclusion

Each major section must contain concise English Markdown explaining:

- what is being done;
- why it is needed;
- important implementation choices;
- expected output.

Avoid excessive tutorial-style explanation.

Use clear English code comments.

Remove unused imports and dead code.

The notebook must be capable of running from the first cell to the last cell in order after a fresh runtime restart.

---

# 15. Required demonstrations

## Test A: exact recipe lookup

```python
ask_cocktail_assistant(
    "How do I make a Mojito?"
)
```

Verify that the result contains:

- Mojito;
- source record;
- ingredients and quantities;
- glass;
- garnish;
- preparation instructions.

## Test B: another named recipe

```python
ask_cocktail_assistant(
    "Give me the recipe for a Negroni."
)
```

## Test C: complete ingredient match

```python
ask_cocktail_assistant(
    "What cocktails can I make?",
    user_ingredients=[
        "white rum",
        "lime juice",
        "mint",
        "simple syrup",
        "soda water"
    ]
)
```

## Test D: incomplete ingredient match

```python
ask_cocktail_assistant(
    "What cocktails can I make with these ingredients?",
    user_ingredients=[
        "gin",
        "lemon juice"
    ]
)
```

The output must list missing required ingredients.

## Test E: refreshing recommendation

```python
ask_cocktail_assistant(
    "Recommend a refreshing cocktail."
)
```

The output must include the heuristic disclosure and score explanation.

## Test F: refreshing recommendation with a constraint

```python
ask_cocktail_assistant(
    "Recommend a refreshing cocktail with gin.",
    user_ingredients=["gin"],
    requested_style="refreshing"
)
```

## Test G: fictional cocktail

```python
ask_cocktail_assistant(
    "How do I make a fictional cocktail called Blue Moon Dragon?"
)
```

The assistant must not invent a recipe.

## Test H: out-of-scope question

```python
ask_cocktail_assistant(
    "Who wrote Hamlet?"
)
```

The assistant must explain that it only handles cocktail recipes and recommendations.

---

# 16. Evaluation

Create a reproducible evaluation set containing at least:

- 5 named-recipe questions;
- 5 ingredient-recommendation cases;
- 5 refreshing-style cases;
- 3 fictional or out-of-scope cases.

Store evaluation cases in a visible Python list or pandas DataFrame.

## 16.1 Retrieval evaluation

Measure:

- exact-title lookup accuracy;
- top-1 semantic retrieval accuracy;
- top-3 semantic retrieval accuracy;
- reciprocal rank where appropriate;
- similarity scores;
- false retrievals;
- threshold rejection behaviour.

Do not evaluate semantic retrieval only with the exact cocktail title.

Include natural question forms such as:

```text
How do I prepare a classic drink made with ...
```

where the expected cocktail is known in advance.

## 16.2 Ingredient-matching evaluation

Test:

- complete matches;
- one missing ingredient;
- multiple missing ingredients;
- aliases;
- optional ingredients;
- pantry defaults;
- recipes with duplicate normalized ingredients;
- zero-ingredient or malformed records.

Verify:

- matched ingredients;
- missing ingredients;
- match ratio;
- ranking order;
- deterministic repeatability.

Use assertions for important calculations.

## 16.3 Refreshing-score evaluation

Create a small manually reviewed set containing:

- clearly refreshing recipes;
- clearly creamy or dessert-style recipes;
- ambiguous recipes.

Record:

- expected broad category;
- calculated score;
- positive signals;
- negative signals;
- whether the result is reasonable.

Discuss misclassifications instead of adjusting weights only to make every example pass.

## 16.4 Generated-answer evaluation

Use a manual rubric with scores from 0 to 2:

- correctness;
- relevance;
- grounding;
- completeness;
- clarity;
- no hallucinated ingredients;
- no hallucinated quantities;
- appropriate uncertainty.

Store results in a pandas DataFrame.

## 16.5 Baseline comparison

Compare:

1. the selected language model without retrieved context;
2. the same model with RAG context;
3. deterministic source formatting where relevant.

Use both well-known and less common cocktails.

Evaluate whether RAG improves:

- source accuracy;
- quantity accuracy;
- completeness;
- hallucination control;
- source traceability.

Do not claim that RAG is better unless the recorded evidence supports the claim.

## 16.6 Efficiency evaluation

Measure:

- dataset loading time;
- cleaning time;
- embedding-generation time;
- FAISS index-construction time;
- average retrieval latency;
- average generation latency;
- full response latency;
- number of indexed recipes;
- embedding dimension;
- model-loading configuration;
- GPU used;
- peak GPU memory where practical.

---

# 17. Evidence required in the notebook

The completed notebook must visibly show:

- raw dataset example;
- parsed ingredient example;
- cleaning summary;
- removed or repaired record counts;
- normalized ingredient examples;
- recipe document example;
- embedding shape;
- FAISS index size;
- exact-title retrieval;
- semantic retrieval;
- similarity scores;
- ingredient-match calculations;
- missing-ingredient calculations;
- refreshing-score calculations;
- language-model answer;
- retrieved source evidence;
- evaluation tables;
- baseline comparison;
- failure cases;
- timing results;
- limitations.

Do not clear all outputs before final submission unless separate screenshots or executed evidence are supplied.

---

# 18. Limitations to discuss

Include at least these limitations:

- the dataset may contain noise, duplicates, missing fields, or inconsistent ingredient wording;
- brand names may complicate normalization;
- the alias dictionary cannot represent every ingredient relationship;
- ingredient-name matching does not verify whether the user has enough quantity;
- some garnishes may be interpreted differently from required ingredients;
- a high ingredient-match score does not guarantee that the user will enjoy the drink;
- refreshing score is a manually designed heuristic, not a source label;
- heuristic weights reflect design judgement;
- semantic retrieval thresholds require empirical validation;
- the language model may still generate unsupported wording;
- deterministic validation can reduce but not eliminate generation risk;
- recipe data should be checked before real-world preparation;
- the system does not provide health or medical advice;
- responsible alcohol use remains the user's responsibility;
- dataset licensing and source limitations must be documented accurately.

Remove the previous limitation about Chinese retrieval because this project is English-only.

---

# 19. Reproducibility files

Create or update:

```text
README.md
requirements.txt
.gitignore
docs/PROJECT_SPEC.md
notebooks/cocktail_rag_assistant.ipynb
```

The README must contain:

- project overview;
- problem statement;
- three supported use cases;
- dataset;
- architecture;
- selected embedding model;
- selected language model;
- installation;
- hardware assumptions;
- how to run the notebook;
- example questions;
- evaluation summary;
- limitations;
- dataset and model licences;
- acknowledgement of GenAI assistance.

The requirements file must contain only required packages and compatible versions.

Likely dependencies include:

```text
datasets
pandas
numpy
sentence-transformers
faiss-cpu
transformers
accelerate
bitsandbytes
torch
```

Do not pin versions blindly. Select a compatible set after checking the actual model requirements.

The `.gitignore` must exclude:

- model cache;
- Hugging Face cache;
- notebook checkpoints;
- temporary FAISS files;
- temporary outputs;
- environment files;
- API keys;
- large downloaded model weights.

Do not commit:

- Hugging Face tokens;
- API keys;
- downloaded model weights;
- private credentials;
- unnecessary generated files.

---

# 20. Coding requirements

Use:

- descriptive function names;
- type hints where they improve clarity;
- docstrings for important functions;
- stable random seeds;
- small reusable functions;
- explicit constants;
- clear validation errors;
- pandas DataFrames for evaluation summaries.

Avoid:

- unnecessary classes;
- deeply nested functions;
- hidden global state;
- duplicated normalization logic;
- duplicated prompt logic;
- hard-coded dataset row indexes;
- broad exception handling;
- silent fallback behaviour;
- pseudo-code in place of working implementation.

Important logic must be testable independently from the language model.

For example, ingredient matching and freshness scoring must run even when the language model is not loaded.

---

# 21. Project completion criteria

The project is complete only when:

1. the notebook runs from top to bottom;
2. the dataset is loaded and cleaned;
3. complete recipe documents are indexed;
4. exact-title lookup works;
5. semantic retrieval works;
6. ingredient recommendations are deterministic;
7. missing ingredients are displayed correctly;
8. refreshing recommendations are explainable;
9. the local language model loads successfully;
10. generated answers are grounded;
11. fictional recipes are rejected;
12. out-of-scope questions are rejected;
13. evaluation results are displayed;
14. baseline comparison is included;
15. README and requirements are accurate;
16. no credentials or large model files are committed;
17. all implementation choices can be explained in a student presentation.

---

# 22. Required working process

Do not implement the full project in one uncontrolled pass.

## Phase 0 閳?inspect and plan

For the first task, do only the following:

1. Inspect the repository structure.
2. Read the reference notebook.
3. Check whether related files already exist.
4. Identify the current Python environment and dependency files.
5. Review this specification for contradictions or technically risky requirements.
6. Produce a file-by-file implementation plan.
7. Propose the final dependency set.
8. Identify the tests needed for each phase.
9. Save this specification as `docs/PROJECT_SPEC.md`.
10. Do not yet build the complete notebook.

Return:

- repository findings;
- proposed architecture;
- files to create or modify;
- major risks;
- phased implementation plan;
- any assumptions made.

Stop after Phase 0 and wait for the next implementation instruction.
