import math
from collections import Counter
import conllu

def read_conllu_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return conllu.parse(f.read())

def extract_ngrams(sentences, n, pos_patterns):
    ngram_counts = Counter()
    token_counts = Counter()
    total_tokens = 0

    for sent in sentences:
        tokens = [(t['form'], t['xpos']) for t in sent if isinstance(t['id'], int)]
        total_tokens += len(tokens)
        for i in range(len(tokens) - n + 1):
            window = tokens[i:i+n]
            seq = tuple(pos for _, pos in window)
            if seq not in pos_patterns:
                continue
            words = tuple(w for w, _ in window)
            for w, _ in window:
                token_counts[w] += 1
            ngram_counts[words] += 1

    return ngram_counts, token_counts, total_tokens

def compute_pmi(ngram_counts, token_counts, total_tokens, freq_thresh=3):
    pmi_scores = {}
    for ngram, count in ngram_counts.items():
        if count < freq_thresh:
            continue
        p_ngram = count / total_tokens
        p_individual = 1
        skip = False
        for word in ngram:
            freq = token_counts.get(word, 0)
            if freq == 0:
                skip = True
                break
            p_individual *= freq / total_tokens
        if skip or p_individual == 0:
            continue  # Avoid zero-division
        pmi = math.log2(p_ngram / p_individual)
        pmi_scores[ngram] = pmi
    return pmi_scores

def identify(pmi_scores, pmi_thresh=5):
    return {ng: sc for ng, sc in pmi_scores.items() if sc > pmi_thresh}

# ---- Bigram classifier using PMI ----
def classify_bigram(ngram, pos_seq, pmi_score):
    # PMI-based classification
    if pmi_score > 8:  # High PMI indicates strong association, likely MWE_R_C
        return 'MWE_R_C'
    elif 5 < pmi_score <= 8:  # Moderate PMI, likely MWE_R_Cl
        return 'MWE_R_Cl'
    elif 3 < pmi_score <= 5:  # Lower PMI, likely MWE_R_E
        return 'MWE_R_E'
    elif pos_seq in [('INTJ', 'INTJ')]:  # Expressive RMWEs
        return 'MWE_R_Ex'
    elif ngram[0] in ('क्या', 'कौन', 'किसी', 'कैसे') and ngram[0] == ngram[1]:  # Wh-type RMWEs
        return 'MWE_R_Wh'
    else:
        return 'MWE_OTH'

# ---- Trigram classifier ----
def classify_trigram(_, pos_seq):
    if pos_seq in (('JJ', 'NN', 'NN'), ('NN', 'NN', 'NN')):
        return 'MWE_CN'
    if pos_seq in (('NN', 'NN', 'VM'), ('JJ', 'NN', 'VM')):
        return 'MWE_LVC'
    return 'MWE_OTH'

def tag_mwes(sentences, mwes, classifier, pmi_scores=None):
    idx = 1
    for sent in sentences:
        forms = [t['form'] for t in sent if isinstance(t['id'], int)]
        xpos = [t['xpos'] for t in sent if isinstance(t['id'], int)]
        for ngram in mwes:
            n = len(ngram)
            for i in range(len(forms) - n + 1):
                if tuple(forms[i:i+n]) == ngram:
                    pos_seq = tuple(xpos[i:i+n])
                    pmi_score = pmi_scores.get(ngram, 0) if pmi_scores else 0
                    tag = classifier(ngram, pos_seq, pmi_score) if pmi_scores else classifier(ngram, pos_seq)
                    for j in range(n):
                        tok = sent[i+j]
                        if tok['misc'] is None:  # Ensure 'misc' is a dictionary
                            tok['misc'] = {}
                        tok['misc']['MWE'] = f"{idx}:{tag}"
                    idx += 1
    return sentences

def write_conllu(sentences, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for s in sentences:
            f.write(s.serialize() + '\n')

# === MAIN ===
in_file = 'combined_unique.conllu'
out_file = 'tagged_output.conllu'
sents = read_conllu_file(in_file)

# Define POS patterns
bigram_pats = {('NN', 'NN'), ('JJ', 'NN'), ('VM', 'NN')}
trigram_pats = {('JJ', 'NN', 'NN'), ('NN', 'NN', 'NN'),
                ('NN', 'NN', 'VM'), ('JJ', 'NN', 'VM')}

# Extract & score bigrams
bi_counts, tok_counts, tot = extract_ngrams(sents, 2, bigram_pats)
bi_pmi = compute_pmi(bi_counts, tok_counts, tot, freq_thresh=3)
bi_mwes = identify(bi_pmi, pmi_thresh=4)

# Extract & score trigrams
tri_counts, tri_tok_counts, _ = extract_ngrams(sents, 3, trigram_pats)
tri_pmi = compute_pmi(tri_counts, tri_tok_counts, tot, freq_thresh=2)
tri_mwes = identify(tri_pmi, pmi_thresh=5)

# Tag both bigrams and trigrams
sents = tag_mwes(sents, bi_mwes, classify_bigram, pmi_scores=bi_pmi)
sents = tag_mwes(sents, tri_mwes, classify_trigram)

# Write the tagged sentences to the output file
write_conllu(sents, out_file)
print(f"Done — bi: {len(bi_mwes)}, tri: {len(tri_mwes)} MWEs tagged → {out_file}")