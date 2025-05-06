import math
from collections import Counter
import conllu

def read_conllu_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return conllu.parse(f.read())

def extract_ngrams(sentences, n=2, pos_patterns=None):
    ngram_counts = Counter()
    token_counts = Counter()
    total_tokens = 0

    for sent in sentences:
        tokens = [(token['form'], token['xpos']) for token in sent if isinstance(token['id'], int)]
        total_tokens += len(tokens)

        for i in range(len(tokens) - n + 1):
            ngram = tokens[i:i + n]
            if pos_patterns:
                pos_sequence = tuple(pos for _, pos in ngram)
                if pos_sequence not in pos_patterns:
                    continue
            words = tuple(word for word, _ in ngram)
            for word in words:
                token_counts[word] += 1
            ngram_counts[words] += 1

    return ngram_counts, token_counts, total_tokens

def compute_pmi(ngram_counts, token_counts, total_tokens, threshold=3):
    pmi_scores = {}
    for ngram, count in ngram_counts.items():
        if count < threshold:
            continue
        p_ngram = count / total_tokens
        p_individual = 1
        for word in ngram:
            p_individual *= token_counts[word] / total_tokens
        pmi = math.log2(p_ngram / p_individual)
        pmi_scores[ngram] = pmi
    return pmi_scores

def identify_mwes(pmi_scores, threshold=5):
    return {ngram: score for ngram, score in pmi_scores.items() if score > threshold}

# ---- Basic Reduplication Classifier (simplified)
def classify_rmwe(word1, word2):
    if word1 == word2:
        return "MWE_R_C"
    elif word1[1:] == word2[1:] or word1[-1:] == word2[-1:]:
        return "MWE_R_E"
    elif word1 in {'क्या', 'कौन', 'कैसे'}:
        return "MWE_R_Wh"
    elif word1 in {'झन', 'झर', 'घूँ', 'सूँ'}:
        return "MWE_R_Ex"
    else:
        return "MWE_R_Cl"

# ---- Annotate MISC Field
def tag_mwes_in_conllu(sentences, mwes):
    mwe_index = 1
    for sent in sentences:
        forms = [token['form'] for token in sent if isinstance(token['id'], int)]
        for ngram in mwes:
            n = len(ngram)
            for i in range(len(forms) - n + 1):
                if tuple(forms[i:i+n]) == ngram:
                    # Get MWE type
                    tag_type = classify_rmwe(*ngram) if len(ngram) == 2 else "MWE_Comp"
                    for j in range(n):
                        token = sent[i + j]
                        if 'misc' not in token or token['misc'] is None:
                            token['misc'] = {}
                        token['misc']['MWE'] = f"{mwe_index}:{tag_type}"
                    mwe_index += 1
    return sentences

def write_conllu_file(sentences, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for sent in sentences:
            f.write(sent.serialize())
            f.write('\n')

# === MAIN EXECUTION ===

conllu_file = 'combined_unique.conllu'
output_file = 'tagged_output.conllu'

sentences = read_conllu_file(conllu_file)

# Use Hindi POS tags (xpos) in patterns
bigram_patterns = {('NN', 'NN'), ('JJ', 'NN'), ('VM', 'NN')}
trigram_patterns = {
    ('JJ', 'NN', 'NN'), 
    ('NN', 'NST', 'NN'),
    ('JJ', 'JJ', 'NN')
}

# Extract, compute PMI and tag
bigram_counts, token_counts, total_tokens = extract_ngrams(sentences, 2, bigram_patterns)
bigram_pmi = compute_pmi(bigram_counts, token_counts, total_tokens)
bigram_mwes = identify_mwes(bigram_pmi, threshold=5)

# Annotate sentences with RMWE MISC tags
tagged_sentences = tag_mwes_in_conllu(sentences, bigram_mwes)

# Output
write_conllu_file(tagged_sentences, output_file)
print(f"Done. Tagged output written to {output_file}")