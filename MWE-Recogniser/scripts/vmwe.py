from conllu import parse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import datetime

# #### takes .cupt file
### change the file name and path accordingly
data_file = open("../data/dev_cause.cupt", "r", encoding="utf-8")
data = data_file.read()
list_tokenlist = parse(data)
for line in list_tokenlist:
    for token in line:
        token["parseme:mwe"] = "*"

va_ending = ['वा', 'वाती', 'वाता', 'वाते', 'वाया', 'वाई', 'वायी', 'वाये', 'वाए', 'वाना', 'वाने', 'वानी']
non_verb = ['हैं', 'है', 'चाह', 'चुक','था','हो', 'रह', 'सक', 'वाला', 'चुका', 'चाहिये', 'चाहिए', 'पा', 'पड़','पड', 'पड़','पड़ेगा', 'चहिए']

# Ensure results directory exists
def ensure_results_directory():
    """Create results directory if it doesn't exist"""
    if not os.path.exists("results"):
        os.makedirs("results")
    return "results"

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
                    if (nt['head'] == token['id']) and (nt['xpos'] == 'NN') and nt['deprel'] == 'compound'and (nt['parseme:mwe']== '*'):
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
            if 'new_id' in token:  # Check if exists first to avoid KeyError
                del token['new_id'] ### removes the new_id column
    return sentences

def parse_mwe_type(tag):
    """Parse a parseme:mwe tag to extract MWE type (LVC or MVC)"""
    if tag == '*' or tag is None:
        return None
    
    # Handle cases with multiple MWE annotations
    if ';' in str(tag):
        parts = str(tag).split(';')
        for part in parts:
            if ':LVC' in part:
                return 'LVC'
            elif ':MVC' in part:
                return 'MVC'
    
    # Handle single MWE annotation
    if ':LVC' in str(tag):
        return 'LVC'
    elif ':MVC' in str(tag):
        return 'MVC'
    
    return None

def is_part_of_mwe(tag):
    """Check if a token is part of any MWE"""
    if tag == '*' or tag is None:
        return False
    return True

def evaluate_metrics(predicted_sentences, gold_file_path):
    """Calculate and save detailed metrics for MWE identification"""
    # Load gold standard data
    with open(gold_file_path, "r", encoding="utf-8") as f:
        gold_data = f.read()
    gold_sentences = parse(gold_data)
    
    # Prepare data structures for evaluation
    y_true_all = []  # 1 if token is part of MWE, 0 otherwise
    y_pred_all = []  # 1 if token is part of MWE, 0 otherwise
    
    y_true_type = []  # None, 'LVC', or 'MVC'
    y_pred_type = []  # None, 'LVC', or 'MVC'
    
    # Collect predictions and true values
    for pred_sent, gold_sent in zip(predicted_sentences, gold_sentences):
        for pred_token, gold_token in zip(pred_sent, gold_sent):
            # Only compare real tokens (skip multiword tokens, comments, etc.)
            if isinstance(pred_token['id'], int) and isinstance(gold_token['id'], int):
                # Binary classification: part of MWE or not
                y_true_all.append(1 if is_part_of_mwe(gold_token['parseme:mwe']) else 0)
                y_pred_all.append(1 if is_part_of_mwe(pred_token['parseme:mwe']) else 0)
                
                # Multi-class classification: MWE type
                y_true_type.append(parse_mwe_type(gold_token['parseme:mwe']))
                y_pred_type.append(parse_mwe_type(pred_token['parseme:mwe']))
    
    # Calculate overall metrics (binary classification: MWE or not)
    accuracy = accuracy_score(y_true_all, y_pred_all)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average='binary')
    
    # Create confusion matrix
    cm = confusion_matrix(y_true_all, y_pred_all)
    
    # Calculate metrics for LVC and MVC separately
    # Convert to numpy arrays for easier filtering
    y_true_type_arr = np.array(y_true_type)
    y_pred_type_arr = np.array(y_pred_type)
    
    # For LVC
    lvc_metrics = calculate_type_metrics(y_true_type_arr, y_pred_type_arr, 'LVC')
    
    # For MVC
    mvc_metrics = calculate_type_metrics(y_true_type_arr, y_pred_type_arr, 'MVC')
    
    # Save all results
    save_metrics(accuracy, precision, recall, f1, cm, lvc_metrics, mvc_metrics)
    
    return {
        'overall': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        },
        'lvc': lvc_metrics,
        'mvc': mvc_metrics
    }

def calculate_type_metrics(y_true_arr, y_pred_arr, mwe_type):
    """Calculate metrics for a specific MWE type (LVC or MVC)"""
    # Create binary arrays for the specific MWE type
    y_true_binary = np.array([1 if t == mwe_type else 0 for t in y_true_arr])
    y_pred_binary = np.array([1 if t == mwe_type else 0 for t in y_pred_arr])
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, average='binary')
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def save_confusion_matrix(cm, filename, title):
    """Save a confusion matrix as an image file"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not MWE', 'MWE'],
                yticklabels=['Not MWE', 'MWE'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_metrics(accuracy, precision, recall, f1, cm, lvc_metrics, mvc_metrics):
    """Save metrics to files in the results folder"""
    results_dir = ensure_results_directory()
    
    # Save overall metrics as text
    with open(os.path.join(results_dir, "overall_metrics.txt"), "w") as f:
        f.write("MULTI-WORD EXPRESSION DETECTION METRICS\n")
        f.write("=====================================\n\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
    
    # Save LVC metrics
    with open(os.path.join(results_dir, "lvc_metrics.txt"), "w") as f:
        f.write("LIGHT VERB CONSTRUCTION (LVC) METRICS\n")
        f.write("=====================================\n\n")
        f.write(f"Accuracy:  {lvc_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {lvc_metrics['precision']:.4f}\n")
        f.write(f"Recall:    {lvc_metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {lvc_metrics['f1']:.4f}\n")
    
    # Save MVC metrics
    with open(os.path.join(results_dir, "mvc_metrics.txt"), "w") as f:
        f.write("MULTI-VERB CONSTRUCTION (MVC) METRICS\n")
        f.write("=====================================\n\n")
        f.write(f"Accuracy:  {mvc_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {mvc_metrics['precision']:.4f}\n")
        f.write(f"Recall:    {mvc_metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {mvc_metrics['f1']:.4f}\n")
    
    # Save combined metrics as CSV for easy comparison
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Overall': [accuracy, precision, recall, f1],
        'LVC': [lvc_metrics['accuracy'], lvc_metrics['precision'], lvc_metrics['recall'], lvc_metrics['f1']],
        'MVC': [mvc_metrics['accuracy'], mvc_metrics['precision'], mvc_metrics['recall'], mvc_metrics['f1']]
    })
    metrics_df.to_csv(os.path.join(results_dir, "all_metrics.csv"), index=False)
    
    # Save confusion matrices as images
    save_confusion_matrix(cm, os.path.join(results_dir, "overall_confusion_matrix.png"), 
                         "Overall MWE Detection Confusion Matrix")
    save_confusion_matrix(lvc_metrics['confusion_matrix'], 
                         os.path.join(results_dir, "lvc_confusion_matrix.png"),
                         "LVC Detection Confusion Matrix")
    save_confusion_matrix(mvc_metrics['confusion_matrix'], 
                         os.path.join(results_dir, "mvc_confusion_matrix.png"),
                         "MVC Detection Confusion Matrix")

def calculate_accuracy(predicted_sentences, gold_file_path):
    """Original accuracy calculation function (keeping for compatibility)"""
    with open(gold_file_path, "r", encoding="utf-8") as f:
        gold_data = f.read()
    gold_sentences = parse(gold_data)

    total = 0
    correct = 0

    for pred_sent, gold_sent in zip(predicted_sentences, gold_sentences):
        for pred_token, gold_token in zip(pred_sent, gold_sent):
            # Only compare real tokens (skip multiword tokens, comments, etc.)
            if isinstance(pred_token['id'], int) and isinstance(gold_token['id'], int):
                total += 1
                if pred_token['parseme:mwe'] == gold_token['parseme:mwe']:
                    correct += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy

# Create README file with metrics explanations
def create_readme():
    readme_content = """# MWE Detection Metrics

This directory contains evaluation metrics for the multi-word expression (MWE) detection model:

## Files:

- `overall_metrics.txt`: Precision, recall, F1-score, and accuracy for all MWE types combined
- `lvc_metrics.txt`: Metrics specific to Light Verb Constructions (LVC)
- `mvc_metrics.txt`: Metrics specific to Multi-Verb Constructions (MVC)
- `all_metrics.csv`: All metrics in a single CSV file for easy comparison
- `overall_confusion_matrix.png`: Confusion matrix for overall MWE detection
- `lvc_confusion_matrix.png`: Confusion matrix for LVC detection
- `mvc_confusion_matrix.png`: Confusion matrix for MVC detection

## Metric Definitions:

- **Accuracy**: Proportion of tokens correctly classified (TP+TN)/(TP+TN+FP+FN)
- **Precision**: Proportion of predicted MWEs that are correct TP/(TP+FP)
- **Recall**: Proportion of actual MWEs that were identified TP/(TP+FN)
- **F1-Score**: Harmonic mean of precision and recall 2*(Precision*Recall)/(Precision+Recall)

Where:
- TP = True Positives (correctly identified MWEs)
- TN = True Negatives (correctly identified non-MWEs)
- FP = False Positives (incorrectly labeled as MWE)
- FN = False Negatives (MWEs that were missed)
"""
    
    results_dir = ensure_results_directory()
    with open(os.path.join(results_dir, "README.md"), "w") as f:
        f.write(readme_content)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python vmwe.py <input_file> <output_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_path = sys.argv[2]

    with open(input_file, "r", encoding="utf-8") as data_file:
        data = data_file.read()
    list_tokenlist = parse(data)
    for line in list_tokenlist:
        for token in line:
            token["parseme:mwe"] = "*"

    # Process data
    print("Step 1: Assigning token IDs...")
    new_sent_list = assign_token_id(list_tokenlist)
    
    print("Step 2: Tagging Light Verb Constructions (LVCs)...")
    sentence_lvc = tag_lvc(new_sent_list)
    
    print("Step 3: Getting parse iterations...")
    lvc_iter_id = get_parse_iteration(sentence_lvc)
    
    print("Step 4: Tagging Multi-Verb Constructions (MVCs)...")
    sentence_mvc = tag_mvc(sentence_lvc, lvc_iter_id)

    print(f"Step 5: Saving annotated data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for sentence in sentence_mvc:
            outfile.writelines(sentence.serialize() + '\n')

    # Evaluate results against gold standard
    gold_path = input_file  # Use input file as gold by default, or change as needed
    print("\nStep 6: Running original accuracy calculation...")
    calculate_accuracy(sentence_mvc, gold_path)
    
    print("\nStep 7: Running comprehensive evaluation...")
    metrics = evaluate_metrics(sentence_mvc, gold_path)
    create_readme()
    
    print("\nOverall Metrics:")
    print(f"Accuracy:  {metrics['overall']['accuracy']:.4f}")
    print(f"Precision: {metrics['overall']['precision']:.4f}")
    print(f"Recall:    {metrics['overall']['recall']:.4f}")
    print(f"F1-Score:  {metrics['overall']['f1']:.4f}")
    
    print("\nLVC Metrics:")
    print(f"Accuracy:  {metrics['lvc']['accuracy']:.4f}")
    print(f"Precision: {metrics['lvc']['precision']:.4f}")
    print(f"Recall:    {metrics['lvc']['recall']:.4f}")
    print(f"F1-Score:  {metrics['lvc']['f1']:.4f}")
    
    print("\nMVC Metrics:")
    print(f"Accuracy:  {metrics['mvc']['accuracy']:.4f}")
    print(f"Precision: {metrics['mvc']['precision']:.4f}")
    print(f"Recall:    {metrics['mvc']['recall']:.4f}")
    print(f"F1-Score:  {metrics['mvc']['f1']:.4f}")
    
    print("\nResults saved to 'results' directory")