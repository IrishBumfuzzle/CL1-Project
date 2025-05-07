# MWE Detection Metrics

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
