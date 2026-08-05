# AI Home Mixologist: Qwen Grounded Generation RAG

An English-language cocktail assistant implemented in one Jupyter notebook. It retrieves recipes from the public `erwanlc/cocktails_recipe` dataset, ranks them with sentence embeddings and FAISS, and uses Qwen only to write a clearly labelled explanation from retrieved evidence.

## Repository contents

| File | Purpose |
| --- | --- |
| `AI_Home_Mixologist_qwen_grounded_generation.ipynb` | The complete implementation, data preparation, retrieval, grounded generation, evaluation, and interactive demo. |
| `requirements.txt` | Python packages needed to reproduce the notebook outside Google Colab. |

There are deliberately no separate Python source files, downloaded datasets, model weights, API keys, or cached indexes in this repository.

## Run the project

### Recommended: Google Colab

1. Upload `AI_Home_Mixologist_qwen_grounded_generation.ipynb` to Google Colab.
2. Select a GPU runtime. Around 12 GB VRAM or more is recommended for the optional 4-bit Qwen model.
3. Run every cell in order, from Section 1 through Section 13.

The first notebook cell installs the required packages. The dataset and model weights are downloaded by the notebook at runtime; they are not stored in this repository.

### Local environment

Use Python 3.10–3.12 and, for Qwen generation, a CUDA-capable GPU.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook AI_Home_Mixologist_qwen_grounded_generation.ipynb
```

On a CPU-only machine, the data preparation, embedding, FAISS retrieval, and deterministic evidence display can run, but the optional Qwen generation may be unavailable or impractical. The notebook displays a deterministic grounded fallback in that case.

## Data preparation and retrieval

The notebook downloads `erwanlc/cocktails_recipe` from Hugging Face, then:

1. inspects the original schema and sample records;
2. removes incomplete or unparsable recipe rows;
3. safely parses ingredient strings with `ast.literal_eval`;
4. normalizes ingredient names and selected aliases for matching;
5. converts each recipe into a searchable document;
6. creates normalized `all-MiniLM-L6-v2` embeddings and a FAISS cosine-similarity index.

No downloaded dataset, embedding cache, FAISS index, model checkpoint, token, or key is committed to GitHub.

## Example commands and questions

Run the notebook cells first, then use its assistant function:

```python
ask_cocktail_assistant("How do I make a Mojito?")
ask_cocktail_assistant("Recommend a cocktail with ginger and lime.")
ask_cocktail_assistant(
    "What can I make?",
    available_ingredients=["gin", "fresh lime", "soda water", "mint"],
)
```

Each result displays the retrieved records or calculated ingredient matches. For semantic recommendations, the notebook shows FAISS cosine similarity and the underlying recipe facts. A weak, fictional, or out-of-scope query is rejected instead of receiving an invented recipe.

## Saved grounded-generation prompt

Section 11 builds the prompt passed to Qwen. Its key constraints are:

> Write 2 to 4 concise English sentences using only the supplied evidence. Do not invent or alter ingredients, quantities, methods, glassware, garnish, origins, rankings, safety warnings, or food-pairing claims. If evidence is insufficient, output `INSUFFICIENT_EVIDENCE`.

The full prompt template, retrieved-record serialization, and response validation are visible in the notebook. Qwen never chooses the retrieval ranking; Python/FAISS produces the evidence first, and the original recipe record remains visible below the generated explanation.

## Evaluation and reproducibility evidence

Section 12 produces live pandas DataFrames for:

- named-recipe retrieval;
- available-ingredient matching;
- semantic-preference retrieval;
- average embedding-plus-FAISS retrieval time;
- final assertions for expected routing and matching behaviour.

Run the notebook from a fresh runtime to regenerate these outputs for the hardware and package versions you use. The notebook intentionally does not claim fixed evaluation scores when the environment has not been executed.

## Models and limitations

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
- Retrieval: FAISS cosine similarity over normalized embeddings.
- Visible language layer: `Qwen/Qwen3-4B-Instruct-2507`, loaded in 4-bit NF4 mode when possible.
- Qwen receives only retrieved recipe records or Python-calculated ingredient results. If loading or generation fails, the notebook retains deterministic retrieved evidence rather than substituting another language model.
- Recipe data can contain duplicates, inconsistent ingredient wording, missing fields, and source noise. Similarity is retrieval evidence, not a verified flavour or cultural claim.
- The project is for recipe information only; check recipes before preparing a drink and consume alcohol responsibly.

