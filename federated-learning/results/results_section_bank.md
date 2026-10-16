### Results — bank



On the bank data (4521 records after removing 0 exact duplicates; 3164/678/679 train/validation/test; majority-class test accuracy 0.885), FedAvg with K=10 IID clients reached a test accuracy of 0.9052 ± 0.0058 (95% CI [0.8980, 0.9123], n=5 seeds), ROC-AUC 0.8843 ± 0.0199 and macro-F1 0.6981 ± 0.0197 at the default 0.5 threshold. The 94% accuracy target was NOT met on this dataset. Standard deviations are across independent seeds (split, partition, initialisation), not across test samples.

The centralized (non-federated) MLP trained on the same records achieved 0.9016 ± 0.0060 accuracy and AUC 0.8846 ± 0.0194; the centralized Random Forest 0.9001 ± 0.0049; and local-only training (mean over the 10 clients) 0.8919 ± 0.0033. Paired differences are in Table 3 (n=5 seeds, so statistical power is limited).

Under label-skewed partitions, iid: accuracy 0.9052 ± 0.0058, AUC 0.8843 ± 0.0199; moderate (Dir 0.5): accuracy 0.8925 ± 0.0043, AUC 0.8808 ± 0.0172; strong (Dir 0.1): accuracy 0.8890 ± 0.0048, AUC 0.8694 ± 0.0240. See Table 4 for training time, communication and client-level spread; robustness should only be claimed where these differences are small relative to the seed SD.

Scalability with the pooled training data held fixed: K=5: acc 0.903 ± 0.004, 1.90 MB, 3.4 s, 28 rounds; K=10: acc 0.905 ± 0.006, 6.46 MB, 5.7 s, 48 rounds; K=20: acc 0.904 ± 0.006, 20.36 MB, 11.0 s, 75 rounds; K=50: acc 0.898 ± 0.002, 54.30 MB, 11.0 s, 80 rounds.