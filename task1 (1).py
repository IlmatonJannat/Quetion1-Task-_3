# Task 3.1: Count the occurrences of words in the combined text and get the top 30 most common words
from collections import Counter
import csv
import re

# Read the combined text file
file_path = '/content/drive/MyDrive/Colab Notebooks/combined_text.txt'
word_counter = Counter()

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Tokenize words in the current line
        words = re.findall(r'\b\w+\b', line.lower())
        # Update word count
        word_counter.update(words)

# Get the top 30 most common words
top_30_words = word_counter.most_common(30)

# Store the top 30 common words and their counts into a CSV file
output_csv_path = '/content/drive/MyDrive/Colab Notebooks/top_30_common_words.csv'
with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Word', 'Count'])
    writer.writerows(top_30_words)

print(f"Top 30 common words saved to: {output_csv_path}")

# Task 3.2: Count unique tokens using the Hugging Face Transformers library
from transformers import AutoTokenizer

# Use the BioBERT tokenizer (or any tokenizer based on your model choice)
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

def get_top_tokens(file_path, top_n=30):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Tokenize the text
    tokens = tokenizer.tokenize(text)

    # Count unique tokens
    token_counter = Counter(tokens)

    # Get the top N tokens
    top_tokens = token_counter.most_common(top_n)

    return top_tokens

# File path of combined text
file_path = '/content/drive/MyDrive/Colab Notebooks/combined_text.txt'
top_tokens = get_top_tokens(file_path)

# Display the top 30 tokens
print("Top 30 Tokens:")
print(top_tokens)

# Task 4: Named-Entity Recognition (NER) using SpaCy/scispaCy and BioBERT
import spacy
from transformers import pipeline

# Load scispaCy models
nlp_en_core_sci_sm = spacy.load("en_core_sci_sm")
nlp_en_ner_bc5cdr_md = spacy.load("en_ner_bc5cdr_md")

# Hugging Face NER pipeline with BioBERT
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
nlp_bert = pipeline("ner", model=model, tokenizer=tokenizer)

# Function to extract diseases and drugs entities using scispaCy models
def extract_entities_scispacy(text, nlp_model):
    doc = nlp_model(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ['DISEASE', 'DRUG']]
    return entities

# Read the combined text
file_path = '/content/drive/MyDrive/Colab Notebooks/combined_text.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Extract entities using en_core_sci_sm
entities_sci_sm = extract_entities_scispacy(text, nlp_en_core_sci_sm)

# Extract entities using en_ner_bc5cdr_md
entities_bc5cdr_md = extract_entities_scispacy(text, nlp_en_ner_bc5cdr_md)

# Extract entities using BioBERT
entities_biobert = nlp_bert(text)

# Compare entities detected by both models
print("Entities detected by en_core_sci_sm:", entities_sci_sm)
print("Entities detected by en_ner_bc5cdr_md:", entities_bc5cdr_md)
print("Entities detected by BioBERT:", entities_biobert)
