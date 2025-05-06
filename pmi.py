import pandas as pd
from collections import Counter
from itertools import combinations
import math
from sklearn.model_selection import train_test_split
from conllu import parse

# Load and preprocess the data
with open('hi_hdtb-ud-train.conllu', 'r', encoding='utf-8') as f:
    conllu_data = parse(f.read())

texts = [" ".join([token['form'] for token in sentence]) for sentence in conllu_data]

# Calculate PMI scores for the dataset

# Extract bigrams considering only NN, NNP (nouns) and ADJ (adjectives)
def extract_valid_bigrams(conllu_data):
    valid_bigrams = []
    valid_trigrams = []
    for sentence in conllu_data:
        tokens = [(token['form'], token['xpostag']) for token in sentence]
        for i in range(len(tokens) - 1):
            word1, tag1 = tokens[i]
            word2, tag2 = tokens[i + 1]
            if (tag1 in ['NN', 'NNP', "NNC", 'ADJ', "JJ"] and tag2 in ['NN', 'NNP', "NNC"]) or (tag1 in ["NN"] and tag2 in ["VM"]):
                valid_bigrams.append((word1, word2))
        for i in range(len(tokens) - 2):
            word1, tag1 = tokens[i]
            word2, tag2 = tokens[i + 1]
            word3, tag3 = tokens[i + 2]
            if (tag1 in ["NN", "NNP", "NNC", "ADJ", "JJ", "VM"] and tag3 in ["NN", "NNP", "NNC", "ADJ", "JJ", "VM"]):
                valid_trigrams.append((word1, word2, word3))
    return valid_bigrams, valid_trigrams

# Extract valid bigrams from the dataset
valid_bigrams, valid_trigrams = extract_valid_bigrams(conllu_data)

# Calculate the frequency of each bigram
bigram_counts = Counter(valid_bigrams)
# Calculate the frequency of each trigram
trigram_counts = Counter(valid_trigrams)

# Output the top 10 most frequent bigrams
print("Top 10 Most Frequent Bigrams (Noun-Noun and Adjective-Noun):")
for bigram, count in bigram_counts.most_common(10):
    print(f"{bigram}: {count}")
print("\nTop 10 Most Frequent Trigrams (Noun-Noun and Adjective-Noun):")
for trigram, count in trigram_counts.most_common(10):
    print(f"{trigram}: {count}")

# Calculate PMI for bigrams and trigrams
def calculate_pmi_for_ngrams(ngrams, word_counts, total_words, threshold=0.01):
    pmi_scores = {}
    for ngram, count in Counter(ngrams).items():
        p_ngram = count / total_words
        # if p_ngram > threshold:
        p_individual = math.prod([word_counts[word] / total_words for word in ngram])
        pmi = math.log2(p_ngram / p_individual)
        pmi_scores[ngram] = pmi
    return pmi_scores

# Calculate word counts and total words for PMI calculation
word_counts = Counter([word for bigram in valid_bigrams for word in bigram] +
                      [word for trigram in valid_trigrams for word in trigram])
total_words = sum(word_counts.values())

mwe_threshold = 1  # Adjust threshold as needed
# Calculate PMI for bigrams and trigrams
bigram_pmi_scores = calculate_pmi_for_ngrams(valid_bigrams, word_counts, total_words, mwe_threshold)
trigram_pmi_scores = calculate_pmi_for_ngrams(valid_trigrams, word_counts, total_words, mwe_threshold)

# Identify MWEs based on PMI threshold
bigram_mwes = [(bigram, score) for bigram, score in bigram_pmi_scores.items() if score > mwe_threshold]
trigram_mwes = [(trigram, score) for trigram, score in trigram_pmi_scores.items() if score > mwe_threshold]

bigram_mwes = sorted(bigram_mwes, key=lambda x: x[1], reverse=True)
trigram_mwes = sorted(trigram_mwes, key=lambda x: x[1], reverse=True)

# Output the identified MWEs
print("\nIdentified Multi-Word Expressions (Bigrams):")
for mwe, score in bigram_mwes:
    print(mwe, score)

print("\nIdentified Multi-Word Expressions (Trigrams):")
for mwe, score in trigram_mwes:
    print(mwe, score)