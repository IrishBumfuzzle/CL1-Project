from conllu import parse
import sys

# print(list_tokenlist)

# #### list of :
# 1. /-va/ endings
# 2. /-a/ endings
# 3. excluded verbs: that do not form /a/ causatives

va_ending = ['वा', 'वाती', 'वाता', 'वाते', 'वाया', 'वाई', 'वायी', 'वाये', 'वाए', 'वाना', 'वाने', 'वानी']
aa_ending = ['ाती', 'ाता', 'ाते', 'ाया', 'ाई', 'ायी', 'ाये', 'ाए', 'ाना', 'ाने', 'ानी']
excluded_verbs = ['बत','पा', 'आ','जा','बता','ला','जता','छा','लुभ','बुल','दोहरा','लुभा','बुला','खटखटा','लहरा','खा']

# #### the function add_cause:
# 1. filters verbs using their xpos
# 2. checks the ending of the verbs
# 3. adds the morph cause annotation to the verbs found in step 2

def add_cause(sentences):
    for sentence in sentences:
        for token in sentence:
            if token['xpos'] == 'VM' or token['xpos']=='VAUX':
                if any(token['form'].endswith(aa) for aa in aa_ending) and (token['lemma'].endswith('ा')) \
                and not (token['lemma'].endswith('ना')) and (token['lemma'] not in excluded_verbs):
                    if 'Cause' not in token['feats']:
                        token['feats']['Cause'] = 'Yes'
                elif any(token['form'].endswith(va) for va in va_ending):
                    if 'Cause' not in token['feats']:
                        token['feats']['Cause'] = 'Yes'
    return sentences

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python feature_cause.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, "r", encoding="utf-8") as data_file:
        data = data_file.read()
    list_tokenlist = parse(data)
    sent_cause = add_cause(list_tokenlist)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for sentence in sent_cause:
            outfile.writelines(sentence.serialize() + '\n')




