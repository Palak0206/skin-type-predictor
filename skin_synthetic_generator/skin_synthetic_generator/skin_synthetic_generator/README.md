# Skin-Type Synthetic Data Generator (Local Ollama)

This project generates synthetic text samples for your skin-type classification
dataset using an LLM running locally through Ollama.

## Expected columns

Your input CSV must contain:

- `text`
- `oiliness`
- `dryness`
- `sensitivity`

Each label must be `Low`, `Medium`, or `High` (case-insensitive).

## 1. Install Ollama

Download and install Ollama: https://ollama.com/

Open a terminal and download a model:

```bash
ollama pull llama3.1:8b
```

Start Ollama (the desktop app usually starts the local service automatically).
Check that it responds:

```bash
ollama list
```

The generator calls `http://localhost:11434` by default. It does not send your
dataset to a hosted API.

## 2. Set up Python

Use Python 3.9 or newer:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 3. Run the generator

Put `generate_skin_data.py` and your CSV in the same folder, then run:

```bash
python generate_skin_data.py --input skin_dataset.csv --target-per-combination 100
```

`100` means: make each label combination already found in your input dataset
have 100 rows in total. It does NOT mean generate 100 additional rows per group.

For example, if a combination currently has 35 rows, the script attempts to
generate 65. If it already has 120, it generates none.

You can choose a different local model:

```bash
python generate_skin_data.py --input skin_dataset.csv --target-per-combination 150 --model llama3.1:8b
```

Useful options:

- `--batch-size 10`: number requested from Ollama per generation call.
- `--max-attempts 8`: retry limit per label combination.
- `--ollama-url http://localhost:11434`: local Ollama server URL.
- `--output-dir synthetic_output`: output folder.

## 4. Output files

The `synthetic_output` folder contains:

- `synthetic_samples.csv`: only newly accepted synthetic rows.
- `augmented_dataset.csv`: original rows followed by synthetic rows.
- `rejected_samples.csv`: rejected candidates and reasons.
- `generation_summary.csv`: original and generated counts by label combination.

The original input CSV is never overwritten.

## Important methodological notes

1. The script balances combinations of all three labels (`oiliness`, `dryness`,
   `sensitivity`) that already exist in your data. It does not invent unseen
   combinations.
2. Exact/normalized text duplicates are rejected against the original data and
   all accepted generated samples. Semantic near-duplicates may still occur.
3. Structural checks cannot guarantee that a sentence truly deserves its
   assigned labels. Manually review generated samples and inspect a random
   sample from every label combination.
4. For honest evaluation, split your original real data into train/validation/
   test first. Generate synthetic samples for the training set only. Do not
   let synthetic or augmented rows enter your held-out test set.
5. Synthetic samples are not new real-world observations. Report their use
   transparently in your project methodology.

## Troubleshooting

- `Connection refused`: open/start Ollama and retry.
- Model not found: run `ollama pull llama3.1:8b` or use a model shown by
  `ollama list`.
- Slow generation: use a smaller model or reduce `--batch-size`.
- Not enough accepted samples: increase `--max-attempts`, inspect the rejected
  CSV, or use a stronger model. Do not lower label validation just to fill counts.
