# 🍜 Soup × Llama 2 / Mistral 7B: Domain Fine-Tuning with Model Soups

Fine-tune **Llama-2-7B** and **Mistral-7B** on a niche domain using the [**Soup CLI**](https://github.com/MakazhanAlpamys/Soup) (`soup-cli`), then blend several fine-tuned runs into one adapter with `soup adapters merge` (a "model soup") and measure the gain.

## What's inside

| File | Purpose |
|------|---------|
| `llm_soup_finetune.ipynb` | Full pipeline: data prep → `soup train` ×N → `soup adapters merge` → evaluation |
| `README.md` | This file |

## How it works

1. **Data**: pulls a domain Q&A dataset from the Hugging Face Hub (default: agriculture) and writes Alpaca-format JSONL (`train` / `val` / `test`).
2. **Ingredients**: generates a `soup.yaml` per run (different learning rate and seed) and trains each with `soup train` using 4-bit QLoRA.
3. **Soups** via `soup adapters merge`:
   - **Uniform soup**: merge every ingredient.
   - **Greedy soup**: add ingredients one by one (best first), keeping each only if validation loss improves ([Wortsman et al., 2022](https://arxiv.org/abs/2203.05482)).
   - Strategy is configurable (`linear`, `ties`, `dare`, `svd`).
4. **Evaluation**: answer-only perplexity on a held-out test set for base, each run and both soups.
5. **Ship**: `soup chat`, `soup push`, `soup merge`, `soup export` (GGUF / Ollama).

> Llama 2 and Mistral have different architectures, so they can't be merged together. Each family gets its own soup.

## Example `soup.yaml` generated per run

```yaml
base: mistralai/Mistral-7B-v0.1
task: sft
data:
  train: work/train.jsonl
  format: alpaca
  val_split: 0.05
training:
  epochs: 1
  lr: 2e-4
  batch_size: auto
  seed: 2
  lora:
    r: 16
    alpha: 32
  quantization: 4bit
output: work/mistral/r2
```

## Requirements

- Python 3.10–3.12
- NVIDIA GPU with ~8–16 GB VRAM for 7B QLoRA (Colab T4 works but is slow). Soup's layer streaming can go lower; see the Soup docs.
- A Hugging Face account/token and access to the gated [`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf).

## Quick start

```bash
git clone https://github.com/NeuralScripter/Soup-Llama-2-Mistral-7B-Domain-Fine-Tuning-with-Model-Soups.git
cd Soup-Llama-2-Mistral-7B-Domain-Fine-Tuning-with-Model-Soups
pip install "soup-cli[train]" jupyter pandas pyyaml
soup doctor                      # check GPU / deps
jupyter notebook llm_soup_finetune.ipynb
```

Or open the notebook in Google Colab with a GPU runtime.

## Using your own domain

Edit the Configuration cell:

```python
DATASET_NAME = "your-org/your-domain-dataset"
QUESTION_COL, ANSWER_COL = "question", "answer"
DOMAIN = "your domain"
```

Good niche candidates: agriculture, regional/national law, traditional medicine, local-language QA, specialised engineering, internal company docs. Check the dataset license and column names.

## Notes

- Soup moves fast. If a CLI flag or YAML key errors, check `soup <command> --help` and the [Soup docs](https://github.com/MakazhanAlpamys/Soup). The relevant commands are `soup train`, `soup adapters merge` and `soup adapters arithmetic`.
- Raise `MAX_SAMPLES` and `EPOCHS`, and add more `RECIPES`, for real results.
- Perplexity measures fit to domain text; evaluate on task metrics or human review before production use.

## Results

Fill in after running:

| Variant | Mistral-7B ppl | Llama-2-7B ppl |
|---------|---------------|----------------|
| Base | | |
| Best single run | | |
| Uniform soup | | |
| Greedy soup | | |

## Suggested `.gitignore`

```
work/
output/
*.safetensors
```

## References

- [Soup CLI](https://github.com/MakazhanAlpamys/Soup)
- Wortsman et al., *Model soups* (2022); Ilharco et al., *Editing models with task arithmetic* (2022)
- Dettmers et al., *QLoRA* (2023)

## License

MIT for this repo's code. Soup is Apache-2.0; base models and datasets keep their own licenses.
