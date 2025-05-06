# MWE_UD_HINDI

This is the README file from the PARSEME verbal multiword expressions (VMWEs) corpus for Hindi updated version submitted to the Joint Workshop on Multiword Expressions and Universal Dependencies (MWE-UD 2024)

## Corpus

This corpus used for this work is downloaded from [PARSEME version 1.3](https://lindat.mff.cuni.cz/repository/xmlui/handle/11372/LRT-5124). Updated files are stored in the not_to_release folder. The data files are divided into three -- dev, train, and test files, as they were in the original corpus.

## Format

The data format follows PARSEME guidelines. All files are in the .cupt format. The following tagsets have been modified for this project:

FEATS (column 6): The tagset is UD features. A new boolean feature Cause=Yes is added to causative verbs.
PARSEME:MWE (column 11): Automatically annotated for following VMWE categories: LVC.cause, LVC.full, MVC.
                         Automatically annotated indirect causatives with feature Cause.
                         Manually annotated for VID and direct causatives.
                         Manually adjudicated for all the VMWE categories and feature Cause.


## Parser
All Python Scripts uses [CONLLU](https://pypi.org/project/conllu/) Parser.

## Python Scripts

All python scripts are in the script folder. Feature 'cause' and VMWEs are annotated separated using different scripts. For our study, we have first annotated the data with cause feature and then used the resultant files as input for annotating VMWEs.

Scripts are uploaded as jupyter notebook.

- feature_cause.ipynb: Automatically annotates indirect causatives with the new feature Cause. 
- vmwe.ipynb: Automatically annotates LVCs and MVCs for Hindi. The script first annotates LVCs using the [compound](https://universaldependencies.org/u/dep/all.html#al-u-dep/compound) deprel from UD framework. It then annotates MVCs.

The two scripts can be used independent of each other as well.
