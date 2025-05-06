from conllu import parse

# #### takes .cupt file

### change the file name and path accordingly
data_file = open("../data/dev_cause.cupt", "r", encoding="utf-8")
# data_file = open("data/dev.cupt", "r", encoding="utf-8") 
data = data_file.read()
list_tokenlist = parse(data)

va_ending = ['वा', 'वाती', 'वाता', 'वाते', 'वाया', 'वाई', 'वायी', 'वाये', 'वाए', 'वाना', 'वाने', 'वानी']
non_verb = ['हैं', 'है', 'चाह', 'चुक','था','हो', 'रह', 'सक', 'वाला', 'चुका', 'चाहिये', 'चाहिए', 'पा', 'पड़','पड', 'पड़','पड़ेगा', 'चहिए']


# #### following functions:
# 1. assign_token_id : assigns sentence id to each token for mapping
# 2. get_parse_iteration: if parseme:mwe column is already tagged this function returns the indexical id
# 3. tag_lvc: annotates LVCs
# 4. tag_mvc: annotates MVCs

def assign_token_id(sentences):
    id_list = []
    for sentence in sentences:
#         print(sentence)
        for token in sentence:
            token['new_id'] = sentence.metadata['source_sent_id']
    return sentences

def get_parse_iteration(sentences):
    parse_dict = {}
    vmwe_iter = {}
    for sentence in sentences:
        for token in sentence:
            if token['new_id'] not in parse_dict:
                parse_dict[token['new_id']] = [token['parseme:mwe']]
            else:
                parse_dict[token['new_id']].append(token['parseme:mwe'])

    for k,v in parse_dict.items():
        l=[int(x) for x in v if isinstance(x,int) or x.isdigit()]
        vmwe_iter[k] = max(l) if l else 0
#     print(lvc_iter)
    return vmwe_iter

def tag_lvc(sentences):
    prev_id = ''
    prev_parse = 0
    for sentence in sentences:
        for token in sentence:
            if token['xpos'] == 'VM':
                next_id = token['new_id']  ### initializing a new variable
                for nt in sentence:  ### nt-> nount_token
                    if (nt['head'] == token['id']) and (nt['xpos'] == 'NN') and nt['deprel'] == 'compound'and (nt['parseme:mwe']=='*'):
                            ### Causative LVCs  -- only va causative
                        if token['form'] !=None and token['feats'] !=None  and any(token['form'].endswith(v) for v in va_ending):
#                         if token['form'] !=None and token['feats'] !=None  and 'Cause' in token['feats']: ## tags all causatives
                            ### checks for if the token belongs to same sentence if yes then increment else start at 1

                            if prev_id == next_id:
                                if token['parseme:mwe'] == '*':
                                    token['parseme:mwe'] = str(int(prev_parse)+1)
                                    nt['parseme:mwe'] = str(token['parseme:mwe'])+':LVC.cause'
                                else:
                                    token['parseme:mwe'] = str(int(prev_parse)+1) + ';'+ token['parseme:mwe']
                                    nt['parseme:mwe'] = str(token['parseme:mwe'])+':LVC.cause'
                                prev_id = next_id
                                prev_parse = int(token['parseme:mwe'])
                                print(prev_parse)
                            
                            else:
                                token['parseme:mwe'] = 1
                                nt['parseme:mwe'] = str(token['parseme:mwe'])+':LVC.cause'
                                prev_parse = int(token['parseme:mwe'])
                                prev_id = next_id
                                
                            prev_parse = int(token['parseme:mwe'])
                            print(prev_parse)
                            prev_id = next_id

                            ### non-causative LVCs
                        else:
                                
                            if prev_id == next_id:
                                if token['parseme:mwe'] == '*':
                                    token['parseme:mwe'] = str(int(prev_parse)+1)
                                    nt['parseme:mwe'] = str(token['parseme:mwe'])+':LVC.full'
                                else:
                                    token['parseme:mwe'] = str(int(prev_parse)+1) + ','+ token['parseme:mwe']
                                    nt['parseme:mwe'] = str(token['parseme:mwe'])+':LVC.full'
                                prev_id = next_id
                                prev_parse = int(token['parseme:mwe'])
                                print(prev_parse)
                            else:
                                token['parseme:mwe'] = 1
                                nt['parseme:mwe'] = str(token['parseme:mwe'])+':LVC.full'
                                prev_parse = int(token['parseme:mwe'])
                                prev_id = next_id
                        prev_parse= int(token['parseme:mwe'])
                        print(prev_parse)
                        print(nt, nt['parseme:mwe'], token,token['parseme:mwe'])
                        # lvc_auto.append((nt['form'], token['form']))
    return sentences

new_sent_list = assign_token_id(list_tokenlist)
sentence_lvc = tag_lvc(new_sent_list)
# for sentence in sentence_lvc:
#     print(sentence.serialize()) ## prints sentences in cupt format

def tag_mvc(sentences, lvc_iter_id):
    prev_id = ''
    prev_parse = 0
    for sentence in sentences:
        for token in sentence:
            if token['xpos']=='VM' and (token['feats']!= None) and 'Aspect' not in token['feats']:
                for vt in sentence: ### vt -> verb token
                    if (vt is not None) and (vt['xpos'] == 'VAUX') and (vt['lemma'] not in non_verb) \
                        and (vt['head'] == token['id']) and (vt['id']-token['id']==1) and vt['feats']!=None:
                        next_id = token['new_id']
                        if 'MVC' in str(token['parseme:mwe']):
                            continue
                        else:
                            for sent_id, iter_no in lvc_iter_id.items():
                                if (vt['new_id'] == sent_id):
                                    if prev_id == next_id:
                                        vt['parseme:mwe'] = prev_parse+1
                                        if token['parseme:mwe'] == '*':
                                            token['parseme:mwe'] = str(vt['parseme:mwe']) + ':MVC.full'
                                        else:
                                            token['parseme:mwe'] = str(token['parseme:mwe'])+';'+str(vt['parseme:mwe'])+':MVC.full'
                                    else:
                                        vt['parseme:mwe'] = iter_no+1
                                        if token['parseme:mwe'] == '*':
                                            token['parseme:mwe'] = str(vt['parseme:mwe']) + ':MVC'
                                        else:
                                            token['parseme:mwe'] = str(token['parseme:mwe'])+';'+str(vt['parseme:mwe'])+':MVC'
                                    prev_id = next_id
                                    prev_parse = vt['parseme:mwe']
            del token['new_id'] ### removes the new_id column
    return sentences

lvc_iter_id = get_parse_iteration(sentence_lvc)
sentence_mvc = tag_mvc(sentence_lvc, lvc_iter_id)

### change file name and path accordingly and save file in cupt format
outfile = open('../data/dev_vmwe.cupt', 'w', encoding='utf-8')

for sentence in sentence_mvc:
#     print(sentence.serialize())
    outfile.writelines(sentence.serialize() + '\n')
outfile.close()




