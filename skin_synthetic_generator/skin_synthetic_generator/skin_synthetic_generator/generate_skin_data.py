#!/usr/bin/env python3
"""
Local LLM synthetic data generator for skin-type text classification.

- Uses Ollama's local HTTP API.
- Balances observed oiliness/dryness/sensitivity label combinations.
- Never edits the input CSV.
- Validates labels and basic text quality.
- Deduplicates against original and newly generated text.
- Exports synthetic-only, augmented, and rejected/audit CSV files.

Example:
  python generate_skin_data.py --input skin_dataset.csv --target-per-combination 100
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

TEXT_COL = "text"

ATTRIBUTE_COLS = [
    "oiliness",
    "dryness",
    "sensitivity"
]

SKIN_TYPE_COL = "skin_type"

LABEL_COLS = ATTRIBUTE_COLS + [SKIN_TYPE_COL]

ATTRIBUTE_LABELS = {
    "low": "Low",
    "medium": "Medium",
    "high": "High"
}

SKIN_TYPE_LABELS = {
    "dry": "Dry",
    "oily": "Oily",
    "normal": "Normal",
    "combination": "Combination"
}


def normalize_text(value):
    """Normalize text only for duplicate comparison; original text stays unchanged."""
    value = str(value).lower().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s']", "", value)
    return value


def canonical_label(value, column):
    key = str(value).strip().lower()

    if column == SKIN_TYPE_COL:
        return SKIN_TYPE_LABELS.get(key)

    return ATTRIBUTE_LABELS.get(key)

def check_dataset(df):
    required = [TEXT_COL] + LABEL_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Expected columns: {required}"
        )

    clean = df.copy()
    clean[TEXT_COL] = clean[TEXT_COL].fillna("").astype(str)

    for col in LABEL_COLS:
        clean[col] = clean[col].map(
            lambda value: canonical_label(value, col)
    )

    invalid = clean[LABEL_COLS].isna().any(axis=1) | (
        clean[TEXT_COL].str.strip() == ""
    )
    if invalid.any():
        examples = df.loc[invalid].head(5)
        raise ValueError(
            f"{int(invalid.sum())} rows have blank text or invalid labels. "
            f"Allowed attribute labels are Low, Medium, High; "
            f"allowed skin types are Dry, Oily, Normal, Combination.\n"
            f"Example invalid rows:\n{examples.to_string(index=False)}"
        )
    return clean


def ollama_chat(base_url, model, prompt, timeout):
    endpoint = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.8 , "top_p": 0.95 },
        "messages": [
            {
                "role": "system",
                "content": (
                    "You create realistic synthetic text-classification examples. "
                    "Follow the requested labels exactly. Return only valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    response = requests.post(endpoint, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    content = data.get("message", {}).get("content", "")
    if not content:
        raise RuntimeError(f"Ollama returned an empty response: {data}")
    return json.loads(content)


def build_prompt(labels, examples, count):

    example_lines = "\n".join(
        f"- {x}" for x in examples
    ) or "- No examples available."

    return f"""
You are generating synthetic data for a skin-type
classification project.

Your task is to write descriptions as if ordinary
people are describing their own skin.

TARGET LABELS:

Oiliness: {labels['oiliness']}
Dryness: {labels['dryness']}
Sensitivity: {labels['sensitivity']}
Skin type: {labels['skin_type']}

WRITING STYLE:

Write like a real person casually explaining
their skin condition.

Examples of the desired writing style:

- my cheeks get pretty dry but my forehead
  and nose are usually really oily
- my skin gets greasy so quickly especially
  around my nose and forehead
- sometimes my face feels dry even though
  my nose gets super oily
- my skin is mostly okay but certain face
  washes make it feel irritated
- around my mouth it gets dry but the rest
  of my face is usually fine

STYLE REQUIREMENTS:

1. Use natural conversational English.
2. Use everyday vocabulary.
3. Allow lowercase sentences.
4. Use casual expressions like kinda, pretty,
   really, usually, sometimes, etc.
5. Allow slightly imperfect grammar.
6. Use different sentence structures.
7. Vary sentence lengths.
8. Mention facial areas naturally.
9. Use first-person descriptions.
10. Avoid medical or textbook terminology.
11. Avoid repetitive sentence structures.
12. Do not make every sentence sound polished
    or artificially generated.

DATA REQUIREMENTS:

1. Generate realistic skin experiences.
2. Every description must match ALL FOUR
   assigned target labels.
3. Preserve the assigned labels exactly.
4. Do not invent contradictory symptoms.
5. Use the existing examples to understand
   the characteristics associated with each
   label combination.
6. Do not copy or lightly paraphrase examples.
7. Do not generate identical descriptions.

EXISTING EXAMPLES FOR REFERENCE:

{example_lines}

Generate exactly {count} NEW descriptions.

Return valid JSON only in this format:

{{
    "samples": [
        {{
            "text": "my skin gets...",
            "oiliness": "{labels['oiliness']}",
            "dryness": "{labels['dryness']}",
            "sensitivity": "{labels['sensitivity']}",
            "skin_type": "{labels['skin_type']}"
        }}
    ]
}}

Do not add explanations outside the JSON.
"""

def main():
    parser = argparse.ArgumentParser(
        description="Generate balanced skin-text samples using a local Ollama model."
    )
    parser.add_argument("--input", required=True, help="Path to original CSV")
    parser.add_argument(
        "--target-per-combination",
        type=int,
        required=True,
        help="Desired final row count for each observed label combination",
    )
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model name")
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434", help="Ollama server URL"
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output-dir", default="synthetic_output")
    args = parser.parse_args()

    if args.target_per_combination < 1 or args.batch_size < 1:
        parser.error("Target and batch size must be positive integers.")

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")

    original_raw = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    original = check_dataset(original_raw)

    # Only the required columns are sent to the generator. Preserve all original
    # columns and values in the augmented output.
    counts = (
        original.groupby(LABEL_COLS, dropna=False)
        .size()
        .rename("current_count")
        .reset_index()
    )

    print("\nCurrent observed label-combination counts:")
    print(counts.to_string(index=False))
    print(
        "\nOnly combinations already present in your dataset will be balanced. "
        "Unseen combinations are not invented automatically."
    )

    seen_text = {
        normalize_text(x)
        for x in original[TEXT_COL].tolist()
        if normalize_text(x)
    }
    generated_rows = []
    rejected_rows = []
    summary_rows = []

    for _, group in counts.iterrows():
        labels = {col: group[col] for col in LABEL_COLS}
        current = int(group["current_count"])
        needed = max(0, args.target_per_combination - current)
        combo_name = " | ".join(f"{k}={v}" for k, v in labels.items())

        print(f"\n{combo_name}: existing={current}, need={needed}")
        if needed == 0:
            summary_rows.append({**labels, "existing": current, "generated": 0})
            continue

        matching = original
        for col, val in labels.items():
            matching = matching[matching[col] == val]
        examples = matching[TEXT_COL].head(8).tolist()

        accepted = 0
        attempts = 0
        while accepted < needed and attempts < args.max_attempts:
            attempts += 1
            request_count = min(args.batch_size, needed - accepted + 3)
            prompt = build_prompt(labels, examples, request_count)

            try:
                result = ollama_chat(
                    args.ollama_url, args.model, prompt, args.timeout
                )
                samples = result.get("samples", [])
                if not isinstance(samples, list):
                    raise ValueError("JSON field 'samples' must be a list.")
            except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
                print(f"  Attempt {attempts} failed: {exc}")
                time.sleep(2)
                continue

            for item in samples:
                if not isinstance(item, dict):
                    rejected_rows.append(
                        {"reason": "not_a_json_object", "raw": str(item)}
                    )
                    continue

                text = str(item.get(TEXT_COL, "")).strip()
                row = {TEXT_COL: text}
                row.update({col: canonical_label(item.get(col, ""), col)for col in LABEL_COLS})
                reason = None
                if not text:
                    reason = "blank_text"
                elif any(row[col] is None for col in LABEL_COLS):
                    reason = "invalid_or_missing_label"
                elif any(row[col] != labels[col] for col in LABEL_COLS):
                    reason = "labels_do_not_match_requested_combination"
                elif len(text.split()) < 4:
                    reason = "text_too_short"
                elif len(text) > 500:
                    reason = "text_too_long"
                else:
                    normalized = normalize_text(text)
                    if normalized in seen_text:
                        reason = "duplicate_text"
                    else:
                        seen_text.add(normalized)

                if reason:
                    rejected_rows.append(
                        {"reason": reason, "text": text, **{
                            col: item.get(col, "") for col in LABEL_COLS
                        }}
                    )
                    continue

                generated_rows.append(row)
                accepted += 1
                if accepted >= needed:
                    break

            print(f"  Accepted {accepted}/{needed} (attempt {attempts})")

        summary_rows.append(
            {**labels, "existing": current, "generated": accepted}
        )
        if accepted < needed:
            print(
                f"  WARNING: short by {needed - accepted} samples. "
                "Try increasing --max-attempts or using a larger Ollama model."
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic = pd.DataFrame(generated_rows, columns=[TEXT_COL] + LABEL_COLS)

    # Preserve original data and all its columns. Generated rows get blank values
    # for any additional columns that exist in the user's original CSV.
    augmented = original_raw.copy()
    for col in augmented.columns:
        if col not in synthetic.columns:
            synthetic[col] = ""
    synthetic = synthetic[augmented.columns]
    augmented = pd.concat([augmented, synthetic], ignore_index=True)

    synthetic_path = output_dir / "synthetic_samples.csv"
    augmented_path = output_dir / "augmented_dataset.csv"
    rejected_path = output_dir / "rejected_samples.csv"
    summary_path = output_dir / "generation_summary.csv"

    synthetic.to_csv(synthetic_path, index=False)
    augmented.to_csv(augmented_path, index=False)
    pd.DataFrame(rejected_rows).to_csv(rejected_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print("\nFinished.")
    print(f"Generated samples: {len(generated_rows)}")
    print(f"Synthetic-only CSV: {synthetic_path}")
    print(f"Augmented CSV:      {augmented_path}")
    print(f"Rejected/audit CSV: {rejected_path}")
    print(f"Summary CSV:        {summary_path}")
    print(
        "\nImportant: review synthetic text and labels before training. "
        "Keep your original held-out validation/test sets untouched; "
        "use synthetic rows in the training set only."
    )


if __name__ == "__main__":
    main()
