import spacy
import pandas as pd
import json  # Added to handle JSON output

# Load SpaCy Catalan model
nlp = spacy.load("ca_core_news_sm")

def process_texts(texts):
    """Tokenize texts and return a set of unique words."""
    words = set()
    for text in texts:
        doc = nlp(text)
        words.update([token.text.lower() for token in doc if token.is_alpha])
    return words

def main(tsv_file):
    # Read TSV file using pandas
    df = pd.read_csv(tsv_file, sep='\t')

    # Extract texts from 'ca' and 'va' columns
    group1_texts = df['ca'].dropna().tolist()
    group2_texts = df['va'].dropna().tolist()

    # Process texts to extract unique words
    group1_words = process_texts(group1_texts)
    group2_words = process_texts(group2_texts)

    # Compute word differences
    only_in_group1 = group1_words - group2_words
    only_in_group2 = group2_words - group1_words

    # Save results to JSON
    discriminative_words = {
        "ca": sorted(list(only_in_group1)),
        "va": sorted(list(only_in_group2))
    }
    with open("data/discriminative.json", "w", encoding="utf-8") as json_file:
        json.dump(discriminative_words, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Replace 'your_file.tsv' with the path to your TSV file
    main("data/CA-VA_alignment_train.tsv")
