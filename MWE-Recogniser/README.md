# MWE-Recogniser

A toolkit for Hindi Multi-Word Expression (MWE) Recognition using statistical measures and rule-based methods.

## Overview

This toolkit provides two main functionalities:
1. **Statistical Analysis**: Calculate PMI (Pointwise Mutual Information) and Chi-Square scores for potential MWEs
2. **Rule-based Tagging**: Identify and tag Light Verb Constructions (LVC) and Multi-Verb Constructions (MVC)

## Prerequisites

- Python 3.6 or higher
- Required Python packages: `conllu`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `sklearn`

## Installation

Clone this repository:

```bash
git clone https://github.com/yourusername/MWE-Recogniser.git
cd MWE-Recogniser
```

## Running the Tool

The toolkit can be run using the provided bash script:

```bash
bash run.sh
```

The script will prompt you to select one of two methods:
1. **PMI and Chi Square Analysis**: This will analyze a CONLLU file using statistical measures
2. **LVC, MVC Tagging**: This will add causative features to your data and then tag Multi-Word Expressions

Follow the prompts to specify input and output files when required.

## Raw Corpus Data

This tool uses Hindi raw corpus data from the PARSEME shared task for training the statistical models. 
**Important Notes**:
- The raw corpus data is **extremely large** (several GB)
- By default, the tool reads only 1000 sentences from each raw corpus file to keep processing manageable
- The raw corpus should be placed in `data/raw-corpus/` directory with files named `raw-001.conllu`, `raw-002.conllu`, etc.
- The first time you run PMI analysis, it will take time to process the corpus files
- Processed bigrams are cached in a pickle file for faster access in subsequent runs

## Methods

### 1. PMI and Chi Square Analysis

Analyzes co-occurrence statistics of word pairs (bigrams) in the text:
- Extracts valid bigrams (Noun-Noun and Adjective-Noun pairs)
- Calculates Point-wise Mutual Information (PMI) scores
- Calculates Chi-Square scores
- Outputs the most statistically significant bigrams that may be MWEs

### 2. LVC and MVC Tagging

A rule-based approach to identify specific types of MWEs:
- First adds causative features to verbs
- Identifies Light Verb Constructions (LVC)
- Identifies Multi-Verb Constructions (MVC)
- Evaluates results against gold standard if available
- Generates detailed metrics and visualizations

## Output

The LVC, MVC tagging generates:
- An annotated CONLLU/CUPT file with MWE tags
- Evaluation metrics (accuracy, precision, recall, F1-score)
- Confusion matrices in the `results` directory
