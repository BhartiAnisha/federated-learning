### Results — hospital



On the hospital data (463 records after removing 236 exact duplicates; 323/70/70 train/validation/test; majority-class test accuracy 0.514), FedAvg with K=10 IID clients reached a test accuracy of 0.9371 ± 0.0239 (95% CI [0.9075, 0.9668], n=5 seeds), ROC-AUC 0.9802 ± 0.0114 and macro-F1 0.9369 ± 0.0241 at the default 0.5 threshold. The 94% accuracy target was NOT met on this dataset. Standard deviations are across independent seeds (split, partition, initialisation), not across test samples.

The centralized (non-federated) MLP trained on the same records achieved 0.9457 ± 0.0212 accuracy and AUC 0.9819 ± 0.0185; the centralized Random Forest 0.9429 ± 0.0286; and local-only training (mean over the 10 clients) 0.9346 ± 0.0224. Paired differences are in Table 3 (n=5 seeds, so statistical power is limited).

Under label-skewed partitions, iid: accuracy 0.9371 ± 0.0239, AUC 0.9802 ± 0.0114; moderate (Dir 0.5): accuracy 0.9457 ± 0.0212, AUC 0.9812 ± 0.0119; strong (Dir 0.1): accuracy 0.9457 ± 0.0235, AUC 0.9755 ± 0.0150. See Table 4 for training time, communication and client-level spread; robustness should only be claimed where these differences are small relative to the seed SD.

Scalability with the pooled training data held fixed: K=5: acc 0.951 ± 0.016, 2.80 MB, 0.6 s, 25 rounds; K=10: acc 0.937 ± 0.024, 7.75 MB, 0.9 s, 35 rounds; K=20: acc 0.940 ± 0.026, 28.54 MB, 1.8 s, 65 rounds; K=50: acc 0.940 ± 0.026, 78.85 MB, 3.8 s, 72 rounds.