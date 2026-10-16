### Results — bank_nodur



On the bank_nodur data (4521 records after removing 0 exact duplicates; 3164/678/679 train/validation/test; majority-class test accuracy 0.885), FedAvg with K=10 IID clients reached a test accuracy of 0.8904 ± 0.0045 (95% CI [0.8849, 0.8960], n=5 seeds), ROC-AUC 0.7297 ± 0.0232 and macro-F1 0.5535 ± 0.0205 at the default 0.5 threshold. The 94% accuracy target was NOT met on this dataset. Standard deviations are across independent seeds (split, partition, initialisation), not across test samples.

The centralized (non-federated) MLP trained on the same records achieved 0.8901 ± 0.0051 accuracy and AUC 0.7320 ± 0.0254; the centralized Random Forest 0.8890 ± 0.0029; and local-only training (mean over the 10 clients) 0.8850 ± 0.0026. Paired differences are in Table 3 (n=5 seeds, so statistical power is limited).

Under label-skewed partitions, iid: accuracy 0.8904 ± 0.0045, AUC 0.7297 ± 0.0232; moderate (Dir 0.5): accuracy 0.8854 ± 0.0038, AUC 0.7223 ± 0.0207; strong (Dir 0.1): accuracy 0.8851 ± 0.0000, AUC 0.6916 ± 0.0297. See Table 4 for training time, communication and client-level spread; robustness should only be claimed where these differences are small relative to the seed SD.

Scalability with the pooled training data held fixed: K=5: acc 0.890 ± 0.004, 4.39 MB, 4.0 s, 33 rounds; K=10: acc 0.890 ± 0.004, 15.77 MB, 7.2 s, 59 rounds; K=20: acc 0.889 ± 0.006, 41.12 MB, 10.1 s, 77 rounds; K=50: acc 0.885 ± 0.002, 106.53 MB, 9.6 s, 80 rounds.