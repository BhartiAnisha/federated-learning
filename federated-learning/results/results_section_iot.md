### Results — iot



On the iot data (37268 records after removing 60337 exact duplicates; 26086/5591/5591 train/validation/test; majority-class test accuracy 0.733), FedAvg with K=10 IID clients reached a test accuracy of 0.9146 ± 0.0044 (95% CI [0.9091, 0.9201], n=5 seeds), ROC-AUC 0.9762 ± 0.0016 and macro-F1 0.8836 ± 0.0072 at the default 0.5 threshold. The 94% accuracy target was NOT met on this dataset. Standard deviations are across independent seeds (split, partition, initialisation), not across test samples.

The centralized (non-federated) MLP trained on the same records achieved 0.9088 ± 0.0051 accuracy and AUC 0.9718 ± 0.0011; the centralized Random Forest 0.9042 ± 0.0043; and local-only training (mean over the 10 clients) 0.8871 ± 0.0051. Paired differences are in Table 3 (n=5 seeds, so statistical power is limited).

Under label-skewed partitions, iid: accuracy 0.9146 ± 0.0044, AUC 0.9762 ± 0.0016; moderate (Dir 0.5): accuracy 0.8996 ± 0.0082, AUC 0.9688 ± 0.0037; strong (Dir 0.1): accuracy 0.8737 ± 0.0232, AUC 0.9501 ± 0.0032. See Table 4 for training time, communication and client-level spread; robustness should only be claimed where these differences are small relative to the seed SD.

Scalability with the pooled training data held fixed: K=5: acc 0.918 ± 0.004, 1.23 MB, 20.8 s, 60 rounds; K=10: acc 0.915 ± 0.004, 2.46 MB, 21.9 s, 60 rounds; K=20: acc 0.909 ± 0.004, 4.92 MB, 22.9 s, 60 rounds; K=50: acc 0.897 ± 0.006, 12.31 MB, 25.5 s, 60 rounds.