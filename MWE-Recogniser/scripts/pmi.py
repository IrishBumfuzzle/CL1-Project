from collections import Counter
import math
from conllu import parse
import os
import pickle
import sys

def extract_valid_bigrams(conllu_data):
    valid_bigrams = []
    valid_trigrams = []
    for sentence in conllu_data:
        tokens = [(token['form'], token['xpostag']) for token in sentence]
        for i in range(len(tokens) - 1):
            word1, tag1 = tokens[i]
            word2, tag2 = tokens[i + 1]
            if (tag1 in ['NN', 'NNP', "NNC", 'ADJ', "JJ"] and tag2 in ['NN', 'NNP', "NNC"]):
                valid_bigrams.append((word1, word2))
    return valid_bigrams, []

def load_or_build_training_bigrams(pickle_path=None):
    if pickle_path is None:
        # Use script directory for pickle file
        pickle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_bigrams.pkl')
    
    if os.path.exists(pickle_path):
        print(f"Loading cached bigrams from {pickle_path}...")
        with open(pickle_path, 'rb') as pf:
            valid_bigrams = pickle.load(pf)
        print(f"Loaded {len(valid_bigrams)} bigrams from cache.")
        return valid_bigrams
    
    print("Building training bigrams from raw corpus files...")
    valid_bigrams = []
    files_processed = 0
    from conllu import parse_incr
    
    for i in range(1, 100):
        # Path to raw corpus files: one level up from script, then into data/raw-corpus/
        file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               'data', 'raw-corpus', f'raw-{str(i).zfill(3)}.conllu')
        if not os.path.exists(file_path):
            continue
            
        print(f"Processing file {i}/99: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            sentence_iter = parse_incr(f)
            count = 0
            batch = []
            for sent in sentence_iter:
                batch.append(sent)
                count += 1
                if count >= 1000:
                    break
            valid_bigrams_temp, _ = extract_valid_bigrams(batch)
            valid_bigrams.extend(valid_bigrams_temp)
            files_processed += 1
    
    print(f"Completed! Processed {files_processed} files, extracted {len(valid_bigrams)} bigrams.")
    print(f"Saving bigrams to {pickle_path} for faster loading next time...")
    with open(pickle_path, 'wb') as pf:
        pickle.dump(valid_bigrams, pf)
    return valid_bigrams

valid_bigrams = load_or_build_training_bigrams()

bigram_counts = Counter(valid_bigrams)

def calculate_chi_square_for_ngrams(ngrams, word_counts, total_words, threshold=3.84):
    chi2_scores = {}
    ngram_counts = Counter(ngrams)
    for ngram, observed in ngram_counts.items():
        expected = 1
        for word in ngram:
            expected *= word_counts[word] / total_words
        expected *= total_words
        if expected > 0:
            chi2 = ((observed - expected) ** 2) / expected
            chi2_scores[ngram] = chi2
    return chi2_scores

def calculate_pmi_for_ngrams(ngrams, word_counts, total_words):
    pmi_scores = {}
    ngram_counts = Counter(ngrams)
    for ngram, observed in ngram_counts.items():
        p_ngram = observed / total_words
        p_words = 1
        for word in ngram:
            p_words *= word_counts[word] / total_words
        if p_ngram > 0 and p_words > 0:
            pmi = math.log2(p_ngram / p_words)
            pmi_scores[ngram] = pmi
    return pmi_scores

word_counts = Counter([word for bigram in valid_bigrams for word in bigram])
total_words = sum(word_counts.values())

chi2_threshold = 3.84

bigram_chi2_scores = calculate_chi_square_for_ngrams(valid_bigrams, word_counts, total_words, chi2_threshold)
bigram_pmi_scores = calculate_pmi_for_ngrams(valid_bigrams, word_counts, total_words)

def output_valid_bigrams_with_scores_from_file(conllu_path, top_k=10):
    with open(conllu_path, 'r', encoding='utf-8') as f:
        conllu_data = parse(f.read())
    valid_bigrams_test, _ = extract_valid_bigrams(conllu_data)
    bigram_counts_test = Counter(valid_bigrams_test)

    print(f"Top {top_k} valid bigrams in {conllu_path}:")
    for bigram, count in bigram_counts_test.most_common(top_k):
        print(f"{bigram}: {count}")

    print(f"\nTop {top_k} valid bigrams by chi-square (using training stats) in {conllu_path}:")
    # Only show chi2 scores for bigrams present in test file, using training stats
    test_bigrams_with_chi2 = [(bigram, bigram_chi2_scores.get(bigram, 0)) for bigram in bigram_counts_test]
    for bigram, score in sorted(test_bigrams_with_chi2, key=lambda x: x[1], reverse=True)[:top_k]:
        print(f"{bigram}: {score:.2f}")

    print(f"\nTop {top_k} valid bigrams by PMI (using training stats) in {conllu_path}:")
    test_bigrams_with_pmi = [(bigram, bigram_pmi_scores.get(bigram, 0)) for bigram in bigram_counts_test]
    for bigram, score in sorted(test_bigrams_with_pmi, key=lambda x: x[1], reverse=True)[:top_k]:
        print(f"{bigram}: {score:.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pmi.py <input_file> [top_k]")
        sys.exit(1)
    input_file = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    output_valid_bigrams_with_scores_from_file(input_file, top_k=top_k)
