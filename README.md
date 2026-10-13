# disvaca: Valencian–Catalan Word Discriminator

This repository contains a small Python library and helper scripts to discriminate between Valencian (va) and Catalan (ca) text segments using discriminative word lists and a SpaCy-based tokenizer.

The core component is the `disvaca` package, which provides a `VACADiscriminator` class that can:

- Detect whether a span of text is more likely Valencian or Catalan
- Count how many words from each variety appear in a text
- Compute ratios of Valencian vs. Catalan words

In addition, the `get_words.py` script can be used to derive discriminative word lists from a parallel Valencian–Catalan corpus.

## What the Project Does

- Tokenizes input text with the SpaCy Catalan model (`ca_core_news_sm`)
- Looks up tokens in language-specific word lists for:
  - Valencian (`va`)
  - Catalan (`ca`)
- Exposes a simple Python API to:
  - Check if a text contains Valencian or Catalan words
  - Get per-language word counts
  - Get per-language word ratios
- Includes utilities to build discriminative word lists from a VA/CA alignment TSV file.

Key components:

- `disvaca/VACADiscriminator`: Discrimination logic and public API
- `disvaca/learned_words.py`: Hand-crafted / learned word lists for `ca` and `va`
- `disvaca/es_vaca.py`: Word lists derived from the `gplsi/es_vaca` dataset (via HuggingFace Datasets)
- `get_words.py`: Script to generate discriminative word lists from [data/CA-VA_alignment_train.tsv](data/CA-VA_alignment_train.tsv) into [data/discriminative.json](data/discriminative.json)

## Why This Project Is Useful

- **Dialect-sensitive processing**: Helps downstream NLP tasks that need to distinguish between Catalan and Valencian variants.
- **Lightweight**: Uses simple word lookup and SpaCy tokenization—no heavy models or training loops.
- **Composable**: The word lists are plain Python/JSON structures that can be inspected, extended, or replaced.
- **Reproducible**: Discriminative lexicons can be regenerated from a TSV alignment corpus or from the `gplsi/es_vaca` dataset.

## Getting Started

### Prerequisites

- Python 3.8+ (tested with recent 3.x versions)
- A working internet connection if you want to load the `gplsi/es_vaca` dataset via HuggingFace Datasets

Install the required Python packages (for the library and helper script):

```bash
pip install spacy pandas datasets
```

Then install the SpaCy Catalan model:

```bash
python -m spacy download ca_core_news_sm
```

> Note: If you are working in a virtual environment (recommended), activate it before running the commands above.

## Usage

### 1. Using the VACADiscriminator in Python

The `VACADiscriminator` class lives in the `disvaca` package and can be instantiated with built-in word lists.

```python
from disvaca import VACADiscriminator

# Load default discriminator (hand-crafted `learned_words` + es_vaca-based lists)
discriminator = VACADiscriminator.load(learned=True, es_vaca=True)

text_va = "Esta vesprada els xiquets juguen al carrer."
text_ca = "Aquesta tarda els nens juguen al carrer."

print(discriminator.is_va(text_va))  # True if VA words detected
print(discriminator.is_ca(text_va))  # True if CA words detected

print(discriminator.get_counts(text_va))   # {'va': n_va_words, 'ca': n_ca_words}
print(discriminator.get_ratios(text_ca))   # {'va': r_va, 'ca': r_ca}

# Get the set of language labels detected in the text
print(discriminator.get_languages(text_va))  # e.g. {'va'}
```

#### Loading with Custom Word Lists

You can provide your own word dictionaries and choose which built-in lists to merge in:

```python
from disvaca import VACADiscriminator

custom_words = {
    "ca": {"hola", "adéu"},
    "va": {"hola", "adeu"},
}

discriminator = VACADiscriminator.load(words=custom_words, learned=True, es_vaca=True)
```

This merges your `custom_words` with:

- The hand-crafted `learned_words` lists
- The lists derived from `gplsi/es_vaca` (if `es_vaca=True`)

### 2. Generating Discriminative Word Lists from TSV

The script [get_words.py](get_words.py) reads a VA/CA alignment TSV file and extracts discriminative words.

By default, it expects:

- Input: [data/CA-VA_alignment_train.tsv](data/CA-VA_alignment_train.tsv)
- Output: [data/discriminative.json](data/discriminative.json)

Run it from the project root:

```bash
python get_words.py
```

What it does:

- Loads the TSV file with columns `ca` and `va`
- Tokenizes each text using the SpaCy Catalan model
- Computes words that are unique to each variety
- Saves the resulting word lists as JSON under the `ca` and `va` keys

You can then inspect or reuse [data/discriminative.json](data/discriminative.json) in your own workflows.

## License

This project is licensed under the Apache License, Version 2.0. See the `LICENSE` file for details.
