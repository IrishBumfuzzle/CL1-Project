# #### takes the cupt file and gives the list of tokens for all the sentences

from conllu import parse

### change the file name and path accordingly
data_file = open("../data/train.cupt", "r", encoding="utf-8") 
data = data_file.read()
list_tokenlist = parse(data)
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

# #### save annotated sentences to a new cupt file

sent_cause = add_cause(list_tokenlist)

### change file name and path
outfile = open('../data/dev_cause.cupt', 'w', encoding='utf-8')

for sentence in sent_cause:
#     print(sentence.serialize())   ## prints annotated data in cupt format
    outfile.writelines(sentence.serialize() + '\n')
outfile.close()




