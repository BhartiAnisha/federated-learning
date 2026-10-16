
# Dataset: bank

## Table 1. Dataset characteristics and client distribution (seed-0 split)
| Item | Value |
|---|---|
| Original shape (rows x cols) | [4521, 17] |
| Exact duplicate records removed | 0 |
| Rows after cleaning | 4521 |
| Raw features / features after encoding | 16 / 51 |
| Classes (count) | {'not subscribed': 4000, 'subscribed': 521} |
| Missing values (original file) | 0 |
| Train / Val / Test | 3164 / 678 / 679 |
| Positive rate train/val/test | 0.115 / 0.115 / 0.115 |
| Majority-class test accuracy (trivial baseline) | 0.8851 |
| Client sizes, K=10, iid | 315–317 (mean 316.4); client positive rate 0.11–0.12 |
| Client sizes, K=10, moderate (Dir 0.5) | 119–857 (mean 316.4); client positive rate 0.01–0.58 |
| Client sizes, K=10, strong (Dir 0.1) | 12–1392 (mean 316.4); client positive rate 0.00–0.99 |

## Table 2. FedAvg (IID, K=10) test performance, threshold=0.5 (default, not tuned), mean ± SD across 5 seeds
| Metric | Mean ± SD (n=5) | Min | Max | 95% CI |
|---|---|---|---|---|
| accuracy | 0.9052 ± 0.0058 | 0.8984 | 0.9131 | [0.8980, 0.9123] |
| precision | 0.6781 ± 0.0574 | 0.6098 | 0.7568 | [0.6068, 0.7493] |
| recall | 0.3359 ± 0.0344 | 0.2821 | 0.3590 | [0.2932, 0.3786] |
| specificity | 0.9790 ± 0.0055 | 0.9734 | 0.9850 | [0.9722, 0.9858] |
| f1 | 0.4481 ± 0.0368 | 0.4000 | 0.4870 | [0.4024, 0.4939] |
| macro_f1 | 0.6981 ± 0.0197 | 0.6736 | 0.7197 | [0.6737, 0.7225] |
| weighted_f1 | 0.8907 ± 0.0066 | 0.8841 | 0.8991 | [0.8824, 0.8989] |
| roc_auc | 0.8843 ± 0.0199 | 0.8535 | 0.9036 | [0.8596, 0.9091] |
| pr_auc | 0.5175 ± 0.0300 | 0.4805 | 0.5511 | [0.4802, 0.5547] |
| balanced_accuracy | 0.6575 ± 0.0170 | 0.6327 | 0.6720 | [0.6364, 0.6786] |
| mcc | 0.4331 ± 0.0384 | 0.3934 | 0.4833 | [0.3854, 0.4809] |
| fpr | 0.0210 ± 0.0055 | 0.0150 | 0.0266 | [0.0142, 0.0278] |
| fnr | 0.6641 ± 0.0344 | 0.6410 | 0.7179 | [0.6214, 0.7068] |
| log_loss | 0.2446 ± 0.0146 | 0.2323 | 0.2687 | [0.2264, 0.2628] |
| brier | 0.0732 ± 0.0030 | 0.0707 | 0.0783 | [0.0694, 0.0770] |

## Table 3. Comparison with centralized, local-only and FL baselines (centralized rows are NOT federated; paired tests on the same seeds/splits, no multiple-comparison correction)
| Method | accuracy | recall | macro_f1 | roc_auc | mcc | Δacc vs FedAvg (FedAvg−X) | paired t p (acc) | d_z (acc) | paired t p (AUC) |
|---|---|---|---|---|---|---|---|---|---|
| FedAvg (IID, K=10) | 0.9052 ± 0.0058 | 0.3359 ± 0.0344 | 0.6981 ± 0.0197 | 0.8843 ± 0.0199 | 0.4331 ± 0.0384 |  |  |  |  |
| Centralized MLP (non-FL) | 0.9016 ± 0.0060 | 0.3385 ± 0.0391 | 0.6934 ± 0.0206 | 0.8846 ± 0.0194 | 0.4167 ± 0.0390 | +0.0035 | 0.186 | 0.71 | 0.836 |
| Centralized Logistic Regression (non-FL) | 0.9040 ± 0.0067 | 0.3128 ± 0.0493 | 0.6872 ± 0.0279 | 0.8808 ± 0.0213 | 0.4173 ± 0.0508 | +0.0012 | 0.700 | 0.18 | 0.077 |
| Centralized sklearn LogReg (non-FL) | 0.9052 ± 0.0066 | 0.3333 ± 0.0462 | 0.6969 ± 0.0242 | 0.8784 ± 0.0206 | 0.4325 ± 0.0435 | +0.0000 | 1.000 | 0.00 | 0.167 |
| Centralized Random Forest (non-FL) | 0.9001 ± 0.0049 | 0.2487 ± 0.0493 | 0.6537 ± 0.0257 | 0.9041 ± 0.0165 | 0.3728 ± 0.0359 | +0.0050 | 0.067 | 1.11 | 0.015 |
| Local-only (mean over clients) | 0.8919 ± 0.0033 | 0.2303 ± 0.0230 | 0.6327 ± 0.0127 | 0.8441 ± 0.0218 | 0.3161 ± 0.0230 | +0.0133 | 0.010 | 2.06 | 0.001 |
| FedProx (mu=0.01, IID) | 0.9046 ± 0.0055 | 0.3385 ± 0.0391 | 0.6980 ± 0.0214 | 0.8846 ± 0.0201 | 0.4306 ± 0.0388 | +0.0006 | 0.178 | 0.73 | 0.221 |

## Table 4. IID vs non-IID (Dirichlet label skew, K=10). Client-level accuracy = global model on each client's own training records, averaged over seeds; it is separate from across-seed SD
| Setting | accuracy | macro_f1 | recall | roc_auc | Train time (s) | Comm (MB) | Rounds | Client-level acc mean (SD across clients) |
|---|---|---|---|---|---|---|---|---|
| iid | 0.9052 ± 0.0058 | 0.6981 ± 0.0197 | 0.3359 ± 0.0344 | 0.8843 ± 0.0199 | 2.09 ± 0.39 | 6.46 ± 1.66 | 47.6 ± 12.3 | 0.9135 (0.0128) |
| moderate (Dir 0.5) | 0.8925 ± 0.0043 | 0.5525 ± 0.0498 | 0.0949 ± 0.0619 | 0.8808 ± 0.0172 | 1.51 ± 0.41 | 4.21 ± 1.34 | 31.0 ± 9.8 | 0.7678 (0.2881) |
| strong (Dir 0.1) | 0.8890 ± 0.0048 | 0.5227 ± 0.0455 | 0.0590 ± 0.0525 | 0.8694 ± 0.0240 | 1.55 ± 0.81 | 4.48 ± 3.03 | 33.0 ± 22.3 | 0.7116 (0.3705) |

## Table 5. Scalability (same pooled training data split across K clients, IID, early stopping on)
| Clients | accuracy | roc_auc | macro_f1 | Rounds | Convergence round | Records/client |
|---|---|---|---|---|---|---|
| 5 | 0.9028 ± 0.0044 | 0.8839 ± 0.0203 | 0.6988 ± 0.0171 | 28.0 ± 8.6 | 18.0 ± 8.6 | 632.8 |
| 10 | 0.9052 ± 0.0058 | 0.8843 ± 0.0199 | 0.6981 ± 0.0197 | 47.6 ± 12.3 | 37.6 ± 12.3 | 316.4 |
| 20 | 0.9043 ± 0.0056 | 0.8844 ± 0.0196 | 0.6984 ± 0.0212 | 75.0 ± 6.9 | 70.0 ± 11.5 | 158.2 |
| 50 | 0.8981 ± 0.0019 | 0.8705 ± 0.0232 | 0.6455 ± 0.0152 | 80.0 ± 0.0 | 80.0 ± 0.0 | 63.3 |

## Table 6. Communication, runtime, memory, model size. Communication = float32 weight payloads actually serialised (4 B/param) x participating clients x rounds, both directions; excludes protocol headers, compression, encryption and retransmissions. Memory = tracemalloc peak of Python/NumPy allocations in the run (not process RSS). Timings are single-process simulation on a shared 2-core CPU, so they are not network latencies.
| Clients | Total time (s) | Avg local (s) | Avg aggregation (ms) | Time/round (s) | Upload (MB) | Download (MB) | Total comm (MB) | Params | Model (MB) | Peak Py/NumPy mem (MB) | Clients/round | Successful/round |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 3.38 ± 0.85 | 0.0225 ± 0.0014 | 0.100 ± 0.006 | 0.122 ± 0.007 | 0.950 ± 0.292 | 0.950 ± 0.292 | 1.901 ± 0.584 | 1697 | 0.00679 | 0.56 ± 0.01 | 5.0 | 5.0 |
| 10 | 5.65 ± 1.58 | 0.0109 ± 0.0004 | 0.149 ± 0.006 | 0.118 ± 0.004 | 3.231 ± 0.832 | 3.231 ± 0.832 | 6.462 ± 1.664 | 1697 | 0.00679 | 0.64 ± 0.01 | 10.0 | 10.0 |
| 20 | 11.01 ± 1.07 | 0.0068 ± 0.0003 | 0.253 ± 0.014 | 0.147 ± 0.007 | 10.182 ± 0.941 | 10.182 ± 0.941 | 20.364 ± 1.881 | 1697 | 0.00679 | 0.83 ± 0.01 | 20.0 | 20.0 |
| 50 | 10.97 ± 0.98 | 0.0025 ± 0.0002 | 0.605 ± 0.114 | 0.137 ± 0.012 | 27.152 ± 0.000 | 27.152 ± 0.000 | 54.304 ± 0.000 | 1697 | 0.00679 | 1.32 ± 0.00 | 50.0 | 50.0 |

## Table 7. Ablations around the tuned configuration (one factor changed at a time; K=10, IID). Client-count ablation is Table 5.
| Variant | accuracy | roc_auc | macro_f1 | Rounds | Comm (MB) | Time (s) |
|---|---|---|---|---|---|---|
| local_epochs=1 | 0.8978 ± 0.0029 | 0.8703 ± 0.0233 | 0.6425 ± 0.0199 | 80.0 ± 0.0 | 10.86 ± 0.00 | 0.97 ± 0.07 |
| local_epochs=3 | 0.9046 ± 0.0061 | 0.8842 ± 0.0190 | 0.6982 ± 0.0193 | 76.6 ± 7.6 | 10.40 ± 1.03 | 2.33 ± 0.46 |
| local_epochs=5 | 0.9052 ± 0.0058 | 0.8843 ± 0.0199 | 0.6981 ± 0.0197 | 47.6 ± 12.3 | 6.46 ± 1.66 | 2.26 ± 0.58 |
| lr=0.009 | 0.9004 ± 0.0065 | 0.8835 ± 0.0197 | 0.6793 ± 0.0209 | 80.0 ± 0.0 | 10.86 ± 0.00 | 3.76 ± 0.29 |
| lr=0.03 | 0.9052 ± 0.0058 | 0.8843 ± 0.0199 | 0.6981 ± 0.0197 | 47.6 ± 12.3 | 6.46 ± 1.66 | 2.29 ± 0.61 |
| lr=0.09 | 0.9013 ± 0.0070 | 0.8845 ± 0.0191 | 0.6892 ± 0.0263 | 23.4 ± 4.4 | 3.18 ± 0.60 | 1.28 ± 0.40 |
| participation=0.2 | 0.9010 ± 0.0077 | 0.8818 ± 0.0155 | 0.6803 ± 0.0396 | 42.0 ± 4.6 | 1.14 ± 0.13 | 0.57 ± 0.12 |
| participation=0.5 | 0.9028 ± 0.0106 | 0.8847 ± 0.0220 | 0.6951 ± 0.0375 | 50.4 ± 19.2 | 3.42 ± 1.30 | 1.39 ± 0.60 |
| participation=1.0 | 0.9052 ± 0.0058 | 0.8843 ± 0.0199 | 0.6981 ± 0.0197 | 47.6 ± 12.3 | 6.46 ± 1.66 | 2.25 ± 0.59 |
| optimizer=sgd | 0.9052 ± 0.0058 | 0.8843 ± 0.0199 | 0.6981 ± 0.0197 | 47.6 ± 12.3 | 6.46 ± 1.66 | 2.34 ± 0.64 |
| optimizer=adam | 0.9019 ± 0.0100 | 0.8788 ± 0.0271 | 0.6854 ± 0.0406 | 13.8 ± 1.8 | 1.87 ± 0.24 | 0.95 ± 0.17 |
| fedprox_mu=0.0 | 0.9052 ± 0.0058 | 0.8843 ± 0.0199 | 0.6981 ± 0.0197 | 47.6 ± 12.3 | 6.46 ± 1.66 | 2.27 ± 0.63 |
| fedprox_mu=0.01 | 0.9046 ± 0.0055 | 0.8846 ± 0.0201 | 0.6980 ± 0.0214 | 48.8 ± 13.5 | 6.63 ± 1.84 | 2.43 ± 0.71 |
| fedprox_mu=0.1 | 0.9046 ± 0.0067 | 0.8841 ± 0.0196 | 0.6971 ± 0.0244 | 51.6 ± 12.7 | 7.01 ± 1.73 | 2.63 ± 0.59 |

## Table 8. Per-seed results and 95% t-based confidence intervals (n=5 seeds)
| seed | accuracy | roc_auc | macro_f1 | mcc | rounds_run | convergence_round |
|---|---|---|---|---|---|---|
| 0 | 0.8984 | 0.8997 | 0.6822 | 0.3934 | 34.0 | 24.0 |
| 1 | 0.9131 | 0.8855 | 0.7197 | 0.4833 | 52.0 | 42.0 |
| 2 | 0.9028 | 0.8535 | 0.6736 | 0.3994 | 38.0 | 28.0 |
| 3 | 0.9028 | 0.9036 | 0.7028 | 0.4305 | 49.0 | 39.0 |
| 4 | 0.9087 | 0.8794 | 0.7123 | 0.4591 | 65.0 | 55.0 |
| mean | 0.9052 | 0.8843 | 0.6981 | 0.4331 |  |  |
| SD (ddof=1) | 0.0058 | 0.0199 | 0.0197 | 0.0384 |  |  |
| 95% CI lo | 0.898 | 0.8596 | 0.6737 | 0.3854 |  |  |
| 95% CI hi | 0.9123 | 0.9091 | 0.7225 | 0.4809 |  |  |

# Dataset: bank_nodur

## Table 1. Dataset characteristics and client distribution (seed-0 split)
| Item | Value |
|---|---|
| Original shape (rows x cols) | [4521, 17] |
| Exact duplicate records removed | 0 |
| Rows after cleaning | 4521 |
| Raw features / features after encoding | 15 / 50 |
| Classes (count) | {'not subscribed': 4000, 'subscribed': 521} |
| Missing values (original file) | 0 |
| Train / Val / Test | 3164 / 678 / 679 |
| Positive rate train/val/test | 0.115 / 0.115 / 0.115 |
| Majority-class test accuracy (trivial baseline) | 0.8851 |
| Client sizes, K=10, iid | 315–317 (mean 316.4); client positive rate 0.11–0.12 |
| Client sizes, K=10, moderate (Dir 0.5) | 119–857 (mean 316.4); client positive rate 0.01–0.58 |
| Client sizes, K=10, strong (Dir 0.1) | 12–1392 (mean 316.4); client positive rate 0.00–0.99 |

## Table 2. FedAvg (IID, K=10) test performance, threshold=0.5 (default, not tuned), mean ± SD across 5 seeds
| Metric | Mean ± SD (n=5) | Min | Max | 95% CI |
|---|---|---|---|---|
| accuracy | 0.8904 ± 0.0045 | 0.8851 | 0.8954 | [0.8849, 0.8960] |
| precision | 0.6654 ± 0.1576 | 0.5000 | 0.8889 | [0.4697, 0.8611] |
| recall | 0.0949 ± 0.0233 | 0.0641 | 0.1154 | [0.0660, 0.1238] |
| specificity | 0.9937 ± 0.0030 | 0.9917 | 0.9983 | [0.9900, 0.9974] |
| f1 | 0.1656 ± 0.0390 | 0.1136 | 0.2000 | [0.1172, 0.2140] |
| macro_f1 | 0.5535 ± 0.0205 | 0.5261 | 0.5716 | [0.5280, 0.5790] |
| weighted_f1 | 0.8522 ± 0.0064 | 0.8438 | 0.8578 | [0.8443, 0.8602] |
| roc_auc | 0.7297 ± 0.0232 | 0.7129 | 0.7700 | [0.7009, 0.7585] |
| pr_auc | 0.3293 ± 0.0475 | 0.2762 | 0.3986 | [0.2703, 0.3882] |
| balanced_accuracy | 0.5443 ± 0.0124 | 0.5279 | 0.5552 | [0.5289, 0.5596] |
| mcc | 0.2219 ± 0.0587 | 0.1477 | 0.2813 | [0.1490, 0.2948] |
| fpr | 0.0063 ± 0.0030 | 0.0017 | 0.0083 | [0.0026, 0.0100] |
| fnr | 0.9051 ± 0.0233 | 0.8846 | 0.9359 | [0.8762, 0.9340] |
| log_loss | 0.3163 ± 0.0093 | 0.3004 | 0.3251 | [0.3047, 0.3279] |
| brier | 0.0908 ± 0.0030 | 0.0863 | 0.0941 | [0.0871, 0.0945] |

## Table 3. Comparison with centralized, local-only and FL baselines (centralized rows are NOT federated; paired tests on the same seeds/splits, no multiple-comparison correction)
| Method | accuracy | recall | macro_f1 | roc_auc | mcc | Δacc vs FedAvg (FedAvg−X) | paired t p (acc) | d_z (acc) | paired t p (AUC) |
|---|---|---|---|---|---|---|---|---|---|
| FedAvg (IID, K=10) | 0.8904 ± 0.0045 | 0.0949 ± 0.0233 | 0.5535 ± 0.0205 | 0.7297 ± 0.0232 | 0.2219 ± 0.0587 |  |  |  |  |
| Centralized MLP (non-FL) | 0.8901 ± 0.0051 | 0.1077 ± 0.0346 | 0.5618 ± 0.0270 | 0.7320 ± 0.0254 | 0.2283 ± 0.0596 | +0.0003 | 0.778 | 0.13 | 0.447 |
| Centralized Logistic Regression (non-FL) | 0.8892 ± 0.0066 | 0.0897 ± 0.0521 | 0.5472 ± 0.0443 | 0.7260 ± 0.0137 | 0.1965 ± 0.1125 | +0.0012 | 0.338 | 0.49 | 0.561 |
| Centralized sklearn LogReg (non-FL) | 0.8901 ± 0.0051 | 0.1256 ± 0.0306 | 0.5739 ± 0.0226 | 0.7232 ± 0.0153 | 0.2422 ± 0.0526 | +0.0003 | 0.828 | 0.10 | 0.286 |
| Centralized Random Forest (non-FL) | 0.8890 ± 0.0029 | 0.1205 ± 0.0431 | 0.5684 ± 0.0284 | 0.7338 ± 0.0359 | 0.2280 ± 0.0414 | +0.0015 | 0.230 | 0.63 | 0.807 |
| Local-only (mean over clients) | 0.8850 ± 0.0026 | 0.0277 ± 0.0135 | 0.4947 ± 0.0128 | 0.6547 ± 0.0182 | 0.0836 ± 0.0486 | +0.0054 | 0.015 | 1.83 | 0.000 |
| FedProx (mu=0.01, IID) | 0.8904 ± 0.0045 | 0.0949 ± 0.0233 | 0.5535 ± 0.0205 | 0.7296 ± 0.0233 | 0.2219 ± 0.0587 | +0.0000 | n/a | n/a | 0.346 |

## Table 4. IID vs non-IID (Dirichlet label skew, K=10). Client-level accuracy = global model on each client's own training records, averaged over seeds; it is separate from across-seed SD
| Setting | accuracy | macro_f1 | recall | roc_auc | Train time (s) | Comm (MB) | Rounds | Client-level acc mean (SD across clients) |
|---|---|---|---|---|---|---|---|---|
| iid | 0.8904 ± 0.0045 | 0.5535 ± 0.0205 | 0.0949 ± 0.0233 | 0.7297 ± 0.0232 | 5.34 ± 1.89 | 15.77 ± 6.02 | 59.2 ± 22.6 | 0.8951 (0.0073) |
| moderate (Dir 0.5) | 0.8854 ± 0.0038 | 0.4843 ± 0.0205 | 0.0154 ± 0.0211 | 0.7223 ± 0.0207 | 3.47 ± 1.41 | 10.28 ± 4.14 | 38.6 ± 15.5 | 0.7521 (0.3165) |
| strong (Dir 0.1) | 0.8851 ± 0.0000 | 0.4791 ± 0.0155 | 0.0103 ± 0.0167 | 0.6916 ± 0.0297 | 2.47 ± 1.01 | 7.78 ± 3.38 | 29.2 ± 12.7 | 0.6990 (0.3973) |

## Table 5. Scalability (same pooled training data split across K clients, IID, early stopping on)
| Clients | accuracy | roc_auc | macro_f1 | Rounds | Convergence round | Records/client |
|---|---|---|---|---|---|---|
| 5 | 0.8901 ± 0.0045 | 0.7300 ± 0.0248 | 0.5530 ± 0.0240 | 33.0 ± 14.5 | 23.0 ± 14.5 | 632.8 |
| 10 | 0.8904 ± 0.0045 | 0.7297 ± 0.0232 | 0.5535 ± 0.0205 | 59.2 ± 22.6 | 52.8 ± 26.9 | 316.4 |
| 20 | 0.8892 ± 0.0060 | 0.7223 ± 0.0197 | 0.5386 ± 0.0326 | 77.2 ± 6.3 | 74.0 ± 10.2 | 158.2 |
| 50 | 0.8854 ± 0.0022 | 0.6781 ± 0.0236 | 0.4771 ± 0.0117 | 80.0 ± 0.0 | 80.0 ± 0.0 | 63.3 |

## Table 6. Communication, runtime, memory, model size. Communication = float32 weight payloads actually serialised (4 B/param) x participating clients x rounds, both directions; excludes protocol headers, compression, encryption and retransmissions. Memory = tracemalloc peak of Python/NumPy allocations in the run (not process RSS). Timings are single-process simulation on a shared 2-core CPU, so they are not network latencies.
| Clients | Total time (s) | Avg local (s) | Avg aggregation (ms) | Time/round (s) | Upload (MB) | Download (MB) | Total comm (MB) | Params | Model (MB) | Peak Py/NumPy mem (MB) | Clients/round | Successful/round |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 4.00 ± 1.41 | 0.0236 ± 0.0105 | 0.111 ± 0.006 | 0.131 ± 0.059 | 2.197 ± 0.964 | 2.197 ± 0.964 | 4.394 ± 1.927 | 3329 | 0.01332 | 1.02 ± 0.01 | 5.0 | 5.0 |
| 10 | 7.15 ± 2.43 | 0.0113 ± 0.0037 | 0.178 ± 0.015 | 0.126 ± 0.041 | 7.883 ± 3.012 | 7.883 ± 3.012 | 15.766 ± 6.024 | 3329 | 0.01332 | 1.18 ± 0.02 | 10.0 | 10.0 |
| 20 | 10.13 ± 1.59 | 0.0060 ± 0.0007 | 0.341 ± 0.059 | 0.131 ± 0.016 | 20.560 ± 1.667 | 20.560 ± 1.667 | 41.120 ± 3.335 | 3329 | 0.01332 | 1.49 ± 0.01 | 20.0 | 20.0 |
| 50 | 9.57 ± 0.76 | 0.0021 ± 0.0002 | 0.711 ± 0.066 | 0.120 ± 0.009 | 53.264 ± 0.000 | 53.264 ± 0.000 | 106.528 ± 0.000 | 3329 | 0.01332 | 2.37 ± 0.00 | 50.0 | 50.0 |

## Table 7. Ablations around the tuned configuration (one factor changed at a time; K=10, IID). Client-count ablation is Table 5.
| Variant | accuracy | roc_auc | macro_f1 | Rounds | Comm (MB) | Time (s) |
|---|---|---|---|---|---|---|
| local_epochs=1 | 0.8884 ± 0.0043 | 0.7048 ± 0.0208 | 0.4993 ± 0.0327 | 80.0 ± 0.0 | 21.31 ± 0.00 | 1.47 ± 0.23 |
| local_epochs=3 | 0.8904 ± 0.0045 | 0.7297 ± 0.0232 | 0.5535 ± 0.0205 | 59.2 ± 22.6 | 15.77 ± 6.02 | 2.65 ± 0.87 |
| local_epochs=5 | 0.8892 ± 0.0049 | 0.7297 ± 0.0241 | 0.5540 ± 0.0246 | 42.8 ± 23.9 | 11.40 ± 6.37 | 2.90 ± 1.56 |
| lr=0.009 | 0.8872 ± 0.0056 | 0.7170 ± 0.0197 | 0.5124 ± 0.0309 | 80.0 ± 0.0 | 21.31 ± 0.00 | 3.31 ± 0.29 |
| lr=0.03 | 0.8904 ± 0.0045 | 0.7297 ± 0.0232 | 0.5535 ± 0.0205 | 59.2 ± 22.6 | 15.77 ± 6.02 | 2.58 ± 0.88 |
| lr=0.09 | 0.8904 ± 0.0046 | 0.7315 ± 0.0240 | 0.5573 ± 0.0167 | 30.4 ± 8.6 | 8.10 ± 2.28 | 1.15 ± 0.30 |
| participation=0.2 | 0.8890 ± 0.0059 | 0.7216 ± 0.0167 | 0.5379 ± 0.0297 | 46.4 ± 9.7 | 2.47 ± 0.52 | 0.50 ± 0.14 |
| participation=0.5 | 0.8898 ± 0.0046 | 0.7310 ± 0.0198 | 0.5532 ± 0.0142 | 59.6 ± 17.9 | 7.94 ± 2.39 | 1.32 ± 0.42 |
| participation=1.0 | 0.8904 ± 0.0045 | 0.7297 ± 0.0232 | 0.5535 ± 0.0205 | 59.2 ± 22.6 | 15.77 ± 6.02 | 2.59 ± 0.94 |
| optimizer=sgd | 0.8904 ± 0.0045 | 0.7297 ± 0.0232 | 0.5535 ± 0.0205 | 59.2 ± 22.6 | 15.77 ± 6.02 | 2.61 ± 1.11 |
| optimizer=adam | 0.8887 ± 0.0066 | 0.7336 ± 0.0263 | 0.5417 ± 0.0394 | 17.8 ± 6.4 | 4.74 ± 1.70 | 1.02 ± 0.44 |
| fedprox_mu=0.0 | 0.8904 ± 0.0045 | 0.7297 ± 0.0232 | 0.5535 ± 0.0205 | 59.2 ± 22.6 | 15.77 ± 6.02 | 2.59 ± 1.26 |
| fedprox_mu=0.01 | 0.8904 ± 0.0045 | 0.7296 ± 0.0233 | 0.5535 ± 0.0205 | 59.2 ± 22.6 | 15.77 ± 6.02 | 2.86 ± 1.34 |
| fedprox_mu=0.1 | 0.8895 ± 0.0052 | 0.7290 ± 0.0233 | 0.5488 ± 0.0262 | 60.0 ± 21.5 | 15.98 ± 5.72 | 2.74 ± 1.13 |

## Table 8. Per-seed results and 95% t-based confidence intervals (n=5 seeds)
| seed | accuracy | roc_auc | macro_f1 | mcc | rounds_run | convergence_round |
|---|---|---|---|---|---|---|
| 0 | 0.8851 | 0.7129 | 0.5261 | 0.1477 | 39.0 | 29.0 |
| 1 | 0.894 | 0.77 | 0.5716 | 0.2672 | 80.0 | 80.0 |
| 2 | 0.8954 | 0.7201 | 0.564 | 0.2813 | 32.0 | 22.0 |
| 3 | 0.8866 | 0.7282 | 0.5371 | 0.1733 | 65.0 | 55.0 |
| 4 | 0.891 | 0.7173 | 0.5686 | 0.2402 | 80.0 | 78.0 |
| mean | 0.8904 | 0.7297 | 0.5535 | 0.2219 |  |  |
| SD (ddof=1) | 0.0045 | 0.0232 | 0.0205 | 0.0587 |  |  |
| 95% CI lo | 0.8849 | 0.7009 | 0.528 | 0.149 |  |  |
| 95% CI hi | 0.896 | 0.7585 | 0.579 | 0.2948 |  |  |

# Dataset: hospital

## Table 1. Dataset characteristics and client distribution (seed-0 split)
| Item | Value |
|---|---|
| Original shape (rows x cols) | [699, 11] |
| Exact duplicate records removed | 236 |
| Rows after cleaning | 463 |
| Raw features / features after encoding | 9 / 9 |
| Classes (count) | {'benign': 225, 'malignant': 238} |
| Missing values (original file) | 16 |
| Train / Val / Test | 323 / 70 / 70 |
| Positive rate train/val/test | 0.514 / 0.514 / 0.514 |
| Majority-class test accuracy (trivial baseline) | 0.5143 |
| Client sizes, K=10, iid | 31–33 (mean 32.3); client positive rate 0.50–0.52 |
| Client sizes, K=10, moderate (Dir 0.5) | 7–63 (mean 32.3); client positive rate 0.03–0.95 |
| Client sizes, K=10, strong (Dir 0.1) | 2–109 (mean 32.3); client positive rate 0.00–1.00 |

## Table 2. FedAvg (IID, K=10) test performance, threshold=0.5 (default, not tuned), mean ± SD across 5 seeds
| Metric | Mean ± SD (n=5) | Min | Max | 95% CI |
|---|---|---|---|---|
| accuracy | 0.9371 ± 0.0239 | 0.9143 | 0.9714 | [0.9075, 0.9668] |
| precision | 0.9321 ± 0.0435 | 0.8571 | 0.9706 | [0.8781, 0.9862] |
| recall | 0.9500 ± 0.0497 | 0.8889 | 1.0000 | [0.8883, 1.0117] |
| specificity | 0.9235 ± 0.0573 | 0.8235 | 0.9706 | [0.8523, 0.9947] |
| f1 | 0.9395 ± 0.0227 | 0.9143 | 0.9730 | [0.9113, 0.9677] |
| macro_f1 | 0.9369 ± 0.0241 | 0.9132 | 0.9713 | [0.9069, 0.9669] |
| weighted_f1 | 0.9370 ± 0.0241 | 0.9134 | 0.9714 | [0.9070, 0.9669] |
| roc_auc | 0.9802 ± 0.0114 | 0.9608 | 0.9886 | [0.9661, 0.9943] |
| pr_auc | 0.9799 ± 0.0122 | 0.9592 | 0.9890 | [0.9647, 0.9951] |
| balanced_accuracy | 0.9368 ± 0.0241 | 0.9118 | 0.9706 | [0.9068, 0.9667] |
| mcc | 0.8775 ± 0.0454 | 0.8301 | 0.9443 | [0.8210, 0.9339] |
| fpr | 0.0765 ± 0.0573 | 0.0294 | 0.1765 | [0.0053, 0.1477] |
| fnr | 0.0500 ± 0.0497 | 0.0000 | 0.1111 | [-0.0117, 0.1117] |
| log_loss | 0.1742 ± 0.0608 | 0.1026 | 0.2664 | [0.0988, 0.2496] |
| brier | 0.0480 ± 0.0162 | 0.0271 | 0.0696 | [0.0279, 0.0681] |

## Table 3. Comparison with centralized, local-only and FL baselines (centralized rows are NOT federated; paired tests on the same seeds/splits, no multiple-comparison correction)
| Method | accuracy | recall | macro_f1 | roc_auc | mcc | Δacc vs FedAvg (FedAvg−X) | paired t p (acc) | d_z (acc) | paired t p (AUC) |
|---|---|---|---|---|---|---|---|---|---|
| FedAvg (IID, K=10) | 0.9371 ± 0.0239 | 0.9500 ± 0.0497 | 0.9369 ± 0.0241 | 0.9802 ± 0.0114 | 0.8775 ± 0.0454 |  |  |  |  |
| Centralized MLP (non-FL) | 0.9457 ± 0.0212 | 0.9611 ± 0.0317 | 0.9455 ± 0.0214 | 0.9819 ± 0.0185 | 0.8930 ± 0.0410 | -0.0086 | 0.374 | -0.45 | 0.700 |
| Centralized Logistic Regression (non-FL) | 0.9457 ± 0.0235 | 0.9611 ± 0.0317 | 0.9455 ± 0.0237 | 0.9802 ± 0.0130 | 0.8929 ± 0.0458 | -0.0086 | 0.426 | -0.40 | 1.000 |
| Centralized sklearn LogReg (non-FL) | 0.9457 ± 0.0186 | 0.9611 ± 0.0373 | 0.9455 ± 0.0188 | 0.9814 ± 0.0142 | 0.8935 ± 0.0360 | -0.0086 | 0.305 | -0.53 | 0.560 |
| Centralized Random Forest (non-FL) | 0.9429 ± 0.0286 | 0.9611 ± 0.0317 | 0.9426 ± 0.0289 | 0.9717 ± 0.0199 | 0.8877 ± 0.0548 | -0.0057 | 0.374 | -0.45 | 0.089 |
| Local-only (mean over clients) | 0.9346 ± 0.0224 | 0.9411 ± 0.0484 | 0.9344 ± 0.0226 | 0.9765 ± 0.0161 | 0.8719 ± 0.0434 | +0.0026 | 0.137 | 0.83 | 0.182 |
| FedProx (mu=0.01, IID) | 0.9371 ± 0.0239 | 0.9500 ± 0.0497 | 0.9369 ± 0.0241 | 0.9802 ± 0.0114 | 0.8775 ± 0.0454 | +0.0000 | n/a | n/a | n/a |

## Table 4. IID vs non-IID (Dirichlet label skew, K=10). Client-level accuracy = global model on each client's own training records, averaged over seeds; it is separate from across-seed SD
| Setting | accuracy | macro_f1 | recall | roc_auc | Train time (s) | Comm (MB) | Rounds | Client-level acc mean (SD across clients) |
|---|---|---|---|---|---|---|---|---|
| iid | 0.9371 ± 0.0239 | 0.9369 ± 0.0241 | 0.9500 ± 0.0497 | 0.9802 ± 0.0114 | 0.30 ± 0.15 | 7.75 ± 3.80 | 35.2 ± 17.3 | 0.9566 (0.0317) |
| moderate (Dir 0.5) | 0.9457 ± 0.0212 | 0.9455 ± 0.0214 | 0.9611 ± 0.0248 | 0.9812 ± 0.0119 | 0.21 ± 0.10 | 5.86 ± 2.89 | 26.6 ± 13.1 | 0.9489 (0.0514) |
| strong (Dir 0.1) | 0.9457 ± 0.0235 | 0.9455 ± 0.0237 | 0.9611 ± 0.0317 | 0.9755 ± 0.0150 | 0.34 ± 0.16 | 8.63 ± 3.92 | 39.2 ± 17.8 | 0.9375 (0.1102) |

## Table 5. Scalability (same pooled training data split across K clients, IID, early stopping on)
| Clients | accuracy | roc_auc | macro_f1 | Rounds | Convergence round | Records/client |
|---|---|---|---|---|---|---|
| 5 | 0.9514 ± 0.0163 | 0.9819 ± 0.0105 | 0.9512 ± 0.0165 | 25.4 ± 9.1 | 15.4 ± 9.1 | 64.6 |
| 10 | 0.9371 ± 0.0239 | 0.9802 ± 0.0114 | 0.9369 ± 0.0241 | 35.2 ± 17.3 | 25.2 ± 17.3 | 32.3 |
| 20 | 0.9400 ± 0.0256 | 0.9809 ± 0.0101 | 0.9397 ± 0.0258 | 64.8 ± 23.0 | 60.4 ± 27.8 | 16.1 |
| 50 | 0.9400 ± 0.0256 | 0.9802 ± 0.0123 | 0.9397 ± 0.0258 | 71.6 ± 18.8 | 69.6 ± 23.3 | 6.5 |

## Table 6. Communication, runtime, memory, model size. Communication = float32 weight payloads actually serialised (4 B/param) x participating clients x rounds, both directions; excludes protocol headers, compression, encryption and retransmissions. Memory = tracemalloc peak of Python/NumPy allocations in the run (not process RSS). Timings are single-process simulation on a shared 2-core CPU, so they are not network latencies.
| Clients | Total time (s) | Avg local (s) | Avg aggregation (ms) | Time/round (s) | Upload (MB) | Download (MB) | Total comm (MB) | Params | Model (MB) | Peak Py/NumPy mem (MB) | Clients/round | Successful/round |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.58 ± 0.18 | 0.0029 ± 0.0004 | 0.099 ± 0.017 | 0.023 ± 0.003 | 1.399 ± 0.500 | 1.399 ± 0.500 | 2.797 ± 0.999 | 2753 | 0.01101 | 0.36 ± 0.01 | 5.0 | 5.0 |
| 10 | 0.89 ± 0.43 | 0.0017 ± 0.0003 | 0.147 ± 0.018 | 0.025 ± 0.004 | 3.876 ± 1.900 | 3.876 ± 1.900 | 7.752 ± 3.800 | 2753 | 0.01101 | 0.48 ± 0.01 | 10.0 | 10.0 |
| 20 | 1.84 ± 0.67 | 0.0010 ± 0.0001 | 0.251 ± 0.026 | 0.028 ± 0.004 | 14.272 ± 5.076 | 14.272 ± 5.076 | 28.543 ± 10.152 | 2753 | 0.01101 | 0.73 ± 0.03 | 20.0 | 20.0 |
| 50 | 3.84 ± 1.19 | 0.0008 ± 0.0001 | 0.618 ± 0.057 | 0.053 ± 0.005 | 39.423 ± 10.342 | 39.423 ± 10.342 | 78.846 ± 20.684 | 2753 | 0.01101 | 1.47 ± 0.04 | 50.0 | 50.0 |

## Table 7. Ablations around the tuned configuration (one factor changed at a time; K=10, IID). Client-count ablation is Table 5.
| Variant | accuracy | roc_auc | macro_f1 | Rounds | Comm (MB) | Time (s) |
|---|---|---|---|---|---|---|
| local_epochs=1 | 0.9371 ± 0.0239 | 0.9802 ± 0.0114 | 0.9369 ± 0.0241 | 35.2 ± 17.3 | 7.75 ± 3.80 | 0.30 ± 0.16 |
| local_epochs=3 | 0.9486 ± 0.0163 | 0.9822 ± 0.0122 | 0.9484 ± 0.0165 | 21.6 ± 11.9 | 4.76 ± 2.62 | 0.38 ± 0.19 |
| local_epochs=5 | 0.9457 ± 0.0212 | 0.9822 ± 0.0151 | 0.9455 ± 0.0214 | 18.4 ± 6.2 | 4.05 ± 1.37 | 0.53 ± 0.27 |
| lr=0.03 | 0.9400 ± 0.0256 | 0.9801 ± 0.0105 | 0.9397 ± 0.0258 | 63.6 ± 22.7 | 14.01 ± 5.01 | 0.57 ± 0.19 |
| lr=0.1 | 0.9371 ± 0.0239 | 0.9802 ± 0.0114 | 0.9369 ± 0.0241 | 35.2 ± 17.3 | 7.75 ± 3.80 | 0.34 ± 0.18 |
| lr=0.3 | 0.9457 ± 0.0212 | 0.9822 ± 0.0113 | 0.9455 ± 0.0214 | 24.2 ± 17.9 | 5.33 ± 3.95 | 0.22 ± 0.15 |
| participation=0.2 | 0.9457 ± 0.0212 | 0.9807 ± 0.0121 | 0.9454 ± 0.0216 | 33.0 ± 14.2 | 1.45 ± 0.63 | 0.13 ± 0.06 |
| participation=0.5 | 0.9400 ± 0.0256 | 0.9812 ± 0.0104 | 0.9397 ± 0.0258 | 37.2 ± 18.9 | 4.10 ± 2.08 | 0.22 ± 0.10 |
| participation=1.0 | 0.9371 ± 0.0239 | 0.9802 ± 0.0114 | 0.9369 ± 0.0241 | 35.2 ± 17.3 | 7.75 ± 3.80 | 0.30 ± 0.12 |
| optimizer=sgd | 0.9371 ± 0.0239 | 0.9802 ± 0.0114 | 0.9369 ± 0.0241 | 35.2 ± 17.3 | 7.75 ± 3.80 | 0.28 ± 0.14 |
| optimizer=adam | 0.9400 ± 0.0256 | 0.9807 ± 0.0159 | 0.9397 ± 0.0258 | 21.8 ± 13.3 | 4.80 ± 2.94 | 0.25 ± 0.13 |
| fedprox_mu=0.0 | 0.9371 ± 0.0239 | 0.9802 ± 0.0114 | 0.9369 ± 0.0241 | 35.2 ± 17.3 | 7.75 ± 3.80 | 0.28 ± 0.12 |
| fedprox_mu=0.01 | 0.9371 ± 0.0239 | 0.9802 ± 0.0114 | 0.9369 ± 0.0241 | 35.2 ± 17.3 | 7.75 ± 3.80 | 0.32 ± 0.14 |
| fedprox_mu=0.1 | 0.9371 ± 0.0239 | 0.9801 ± 0.0117 | 0.9369 ± 0.0241 | 35.2 ± 17.3 | 7.75 ± 3.80 | 0.32 ± 0.14 |

## Table 8. Per-seed results and 95% t-based confidence intervals (n=5 seeds)
| seed | accuracy | roc_auc | macro_f1 | mcc | rounds_run | convergence_round |
|---|---|---|---|---|---|---|
| 0 | 0.9143 | 0.9608 | 0.9132 | 0.8402 | 38.0 | 28.0 |
| 1 | 0.9714 | 0.9886 | 0.9713 | 0.9443 | 38.0 | 28.0 |
| 2 | 0.9429 | 0.9861 | 0.9428 | 0.8856 | 27.0 | 17.0 |
| 3 | 0.9143 | 0.9796 | 0.9143 | 0.8301 | 13.0 | 3.0 |
| 4 | 0.9429 | 0.9861 | 0.9429 | 0.8873 | 60.0 | 50.0 |
| mean | 0.9371 | 0.9802 | 0.9369 | 0.8775 |  |  |
| SD (ddof=1) | 0.0239 | 0.0114 | 0.0241 | 0.0454 |  |  |
| 95% CI lo | 0.9075 | 0.9661 | 0.9069 | 0.821 |  |  |
| 95% CI hi | 0.9668 | 0.9943 | 0.9669 | 0.9339 |  |  |

# Dataset: iot

## Table 1. Dataset characteristics and client distribution (seed-0 split)
| Item | Value |
|---|---|
| Original shape (rows x cols) | [97606, 5] |
| Exact duplicate records removed | 60337 |
| Rows after cleaning | 37268 |
| Raw features / features after encoding | 6 / 6 |
| Classes (count) | {'Out': 27326, 'In': 9942} |
| Missing values (original file) | 0 |
| Train / Val / Test | 26086 / 5591 / 5591 |
| Positive rate train/val/test | 0.267 / 0.267 / 0.267 |
| Majority-class test accuracy (trivial baseline) | 0.7331 |
| Client sizes, K=10, iid | 2607–2609 (mean 2608.6); client positive rate 0.27–0.27 |
| Client sizes, K=10, moderate (Dir 0.5) | 932–5352 (mean 2608.6); client positive rate 0.01–0.93 |
| Client sizes, K=10, strong (Dir 0.1) | 5–11988 (mean 2608.6); client positive rate 0.00–1.00 |

## Table 2. FedAvg (IID, K=10) test performance, threshold=0.5 (default, not tuned), mean ± SD across 5 seeds
| Metric | Mean ± SD (n=5) | Min | Max | 95% CI |
|---|---|---|---|---|
| accuracy | 0.9146 ± 0.0044 | 0.9084 | 0.9186 | [0.9091, 0.9201] |
| precision | 0.9170 ± 0.0106 | 0.9010 | 0.9287 | [0.9038, 0.9301] |
| recall | 0.7479 ± 0.0237 | 0.7158 | 0.7808 | [0.7184, 0.7773] |
| specificity | 0.9753 ± 0.0041 | 0.9688 | 0.9790 | [0.9702, 0.9803] |
| f1 | 0.8236 ± 0.0119 | 0.8066 | 0.8366 | [0.8088, 0.8383] |
| macro_f1 | 0.8836 ± 0.0072 | 0.8733 | 0.8912 | [0.8746, 0.8926] |
| weighted_f1 | 0.9116 ± 0.0051 | 0.9044 | 0.9167 | [0.9053, 0.9179] |
| roc_auc | 0.9762 ± 0.0016 | 0.9744 | 0.9778 | [0.9743, 0.9782] |
| pr_auc | 0.9421 ± 0.0042 | 0.9379 | 0.9464 | [0.9369, 0.9473] |
| balanced_accuracy | 0.8616 ± 0.0104 | 0.8472 | 0.8748 | [0.8487, 0.8744] |
| mcc | 0.7751 ± 0.0121 | 0.7584 | 0.7864 | [0.7601, 0.7902] |
| fpr | 0.0247 ± 0.0041 | 0.0210 | 0.0312 | [0.0197, 0.0298] |
| fnr | 0.2521 ± 0.0237 | 0.2192 | 0.2842 | [0.2227, 0.2816] |
| log_loss | 0.1677 ± 0.0044 | 0.1626 | 0.1722 | [0.1623, 0.1732] |
| brier | 0.0554 ± 0.0018 | 0.0537 | 0.0574 | [0.0532, 0.0577] |

## Table 3. Comparison with centralized, local-only and FL baselines (centralized rows are NOT federated; paired tests on the same seeds/splits, no multiple-comparison correction)
| Method | accuracy | recall | macro_f1 | roc_auc | mcc | Δacc vs FedAvg (FedAvg−X) | paired t p (acc) | d_z (acc) | paired t p (AUC) |
|---|---|---|---|---|---|---|---|---|---|
| FedAvg (IID, K=10) | 0.9146 ± 0.0044 | 0.7479 ± 0.0237 | 0.8836 ± 0.0072 | 0.9762 ± 0.0016 | 0.7751 ± 0.0121 |  |  |  |  |
| Centralized MLP (non-FL) | 0.9088 ± 0.0051 | 0.7387 ± 0.0397 | 0.8757 ± 0.0093 | 0.9718 ± 0.0011 | 0.7597 ± 0.0139 | +0.0058 | 0.009 | 2.14 | 0.001 |
| Centralized Logistic Regression (non-FL) | 0.7997 ± 0.0067 | 0.5712 ± 0.0285 | 0.7346 ± 0.0100 | 0.8831 ± 0.0041 | 0.4715 ± 0.0191 | +0.1149 | 0.000 | 19.11 | 0.000 |
| Centralized sklearn LogReg (non-FL) | 0.8001 ± 0.0053 | 0.5768 ± 0.0168 | 0.7362 ± 0.0076 | 0.8840 ± 0.0038 | 0.4740 ± 0.0148 | +0.1144 | 0.000 | 24.11 | 0.000 |
| Centralized Random Forest (non-FL) | 0.9042 ± 0.0043 | 0.8141 ± 0.0157 | 0.8771 ± 0.0060 | 0.9752 ± 0.0014 | 0.7542 ± 0.0119 | +0.0104 | 0.000 | 18.34 | 0.008 |
| Local-only (mean over clients) | 0.8871 ± 0.0051 | 0.6896 ± 0.0111 | 0.8454 ± 0.0067 | 0.9582 ± 0.0022 | 0.7007 ± 0.0139 | +0.0274 | 0.000 | 7.64 | 0.000 |
| FedProx (mu=0.01, IID) | 0.9133 ± 0.0047 | 0.7430 ± 0.0199 | 0.8817 ± 0.0073 | 0.9751 ± 0.0020 | 0.7716 ± 0.0130 | +0.0013 | 0.186 | 0.71 | 0.005 |

## Table 4. IID vs non-IID (Dirichlet label skew, K=10). Client-level accuracy = global model on each client's own training records, averaged over seeds; it is separate from across-seed SD
| Setting | accuracy | macro_f1 | recall | roc_auc | Train time (s) | Comm (MB) | Rounds | Client-level acc mean (SD across clients) |
|---|---|---|---|---|---|---|---|---|
| iid | 0.9146 ± 0.0044 | 0.8836 ± 0.0072 | 0.7479 ± 0.0237 | 0.9762 ± 0.0016 | 7.86 ± 1.21 | 2.46 ± 0.00 | 60.0 ± 0.0 | 0.9180 (0.0036) |
| moderate (Dir 0.5) | 0.8996 ± 0.0082 | 0.8557 ± 0.0144 | 0.6544 ± 0.0367 | 0.9688 ± 0.0037 | 5.27 ± 2.47 | 1.63 ± 0.72 | 39.8 ± 17.5 | 0.8578 (0.1177) |
| strong (Dir 0.1) | 0.8737 ± 0.0232 | 0.8277 ± 0.0088 | 0.6794 ± 0.1596 | 0.9501 ± 0.0032 | 4.29 ± 1.93 | 1.34 ± 0.56 | 32.6 ± 13.7 | 0.8595 (0.1343) |

## Table 5. Scalability (same pooled training data split across K clients, IID, early stopping on)
| Clients | accuracy | roc_auc | macro_f1 | Rounds | Convergence round | Records/client |
|---|---|---|---|---|---|---|
| 5 | 0.9180 ± 0.0045 | 0.9781 ± 0.0016 | 0.8891 ± 0.0059 | 60.0 ± 0.0 | 59.6 ± 0.5 | 5217.2 |
| 10 | 0.9146 ± 0.0044 | 0.9762 ± 0.0016 | 0.8836 ± 0.0072 | 60.0 ± 0.0 | 58.8 ± 1.1 | 2608.6 |
| 20 | 0.9088 ± 0.0038 | 0.9722 ± 0.0021 | 0.8747 ± 0.0063 | 60.0 ± 0.0 | 59.4 ± 0.5 | 1304.3 |
| 50 | 0.8970 ± 0.0058 | 0.9643 ± 0.0020 | 0.8557 ± 0.0101 | 60.0 ± 0.0 | 60.0 ± 0.0 | 521.7 |

## Table 6. Communication, runtime, memory, model size. Communication = float32 weight payloads actually serialised (4 B/param) x participating clients x rounds, both directions; excludes protocol headers, compression, encryption and retransmissions. Memory = tracemalloc peak of Python/NumPy allocations in the run (not process RSS). Timings are single-process simulation on a shared 2-core CPU, so they are not network latencies.
| Clients | Total time (s) | Avg local (s) | Avg aggregation (ms) | Time/round (s) | Upload (MB) | Download (MB) | Total comm (MB) | Params | Model (MB) | Peak Py/NumPy mem (MB) | Clients/round | Successful/round |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 20.80 ± 1.66 | 0.0665 ± 0.0054 | 0.099 ± 0.008 | 0.347 ± 0.028 | 0.616 ± 0.000 | 0.616 ± 0.000 | 1.231 ± 0.000 | 513 | 0.00205 | 5.89 ± 0.00 | 5.0 | 5.0 |
| 10 | 21.86 ± 1.46 | 0.0350 ± 0.0024 | 0.142 ± 0.007 | 0.364 ± 0.024 | 1.231 ± 0.000 | 1.231 ± 0.000 | 2.462 ± 0.000 | 513 | 0.00205 | 5.92 ± 0.00 | 10.0 | 10.0 |
| 20 | 22.87 ± 1.55 | 0.0183 ± 0.0013 | 0.235 ± 0.023 | 0.381 ± 0.026 | 2.462 ± 0.000 | 2.462 ± 0.000 | 4.925 ± 0.000 | 513 | 0.00205 | 5.98 ± 0.00 | 20.0 | 20.0 |
| 50 | 25.45 ± 1.21 | 0.0081 ± 0.0004 | 0.515 ± 0.012 | 0.424 ± 0.020 | 6.156 ± 0.000 | 6.156 ± 0.000 | 12.312 ± 0.000 | 513 | 0.00205 | 6.17 ± 0.00 | 50.0 | 50.0 |

## Table 7. Ablations around the tuned configuration (one factor changed at a time; K=10, IID). Client-count ablation is Table 5.
| Variant | accuracy | roc_auc | macro_f1 | Rounds | Comm (MB) | Time (s) |
|---|---|---|---|---|---|---|
| local_epochs=1 | 0.9027 ± 0.0029 | 0.9680 ± 0.0019 | 0.8666 ± 0.0045 | 60.0 ± 0.0 | 2.46 ± 0.00 | 3.09 ± 0.22 |
| local_epochs=3 | 0.9146 ± 0.0044 | 0.9762 ± 0.0016 | 0.8836 ± 0.0072 | 60.0 ± 0.0 | 2.46 ± 0.00 | 8.25 ± 0.68 |
| local_epochs=5 | 0.9181 ± 0.0041 | 0.9781 ± 0.0012 | 0.8898 ± 0.0058 | 60.0 ± 0.0 | 2.46 ± 0.00 | 13.30 ± 0.82 |
| lr=0.09 | 0.9079 ± 0.0049 | 0.9707 ± 0.0019 | 0.8734 ± 0.0078 | 60.0 ± 0.0 | 2.46 ± 0.00 | 7.75 ± 0.46 |
| lr=0.3 | 0.9146 ± 0.0044 | 0.9762 ± 0.0016 | 0.8836 ± 0.0072 | 60.0 ± 0.0 | 2.46 ± 0.00 | 7.89 ± 0.29 |
| lr=0.9 | 0.9043 ± 0.0085 | 0.9696 ± 0.0052 | 0.8673 ± 0.0148 | 60.0 ± 0.0 | 2.46 ± 0.00 | 7.75 ± 0.60 |
| participation=0.2 | 0.9125 ± 0.0048 | 0.9748 ± 0.0022 | 0.8812 ± 0.0070 | 59.0 ± 1.4 | 0.48 ± 0.01 | 2.10 ± 0.30 |
| participation=0.5 | 0.9140 ± 0.0043 | 0.9759 ± 0.0017 | 0.8828 ± 0.0056 | 60.0 ± 0.0 | 1.23 ± 0.00 | 4.29 ± 0.44 |
| participation=1.0 | 0.9146 ± 0.0044 | 0.9762 ± 0.0016 | 0.8836 ± 0.0072 | 60.0 ± 0.0 | 2.46 ± 0.00 | 7.84 ± 0.34 |
| optimizer=sgd | 0.9146 ± 0.0044 | 0.9762 ± 0.0016 | 0.8836 ± 0.0072 | 60.0 ± 0.0 | 2.46 ± 0.00 | 7.82 ± 0.62 |
| optimizer=adam | 0.9106 ± 0.0033 | 0.9733 ± 0.0015 | 0.8769 ± 0.0055 | 60.0 ± 0.0 | 2.46 ± 0.00 | 10.42 ± 1.28 |
| fedprox_mu=0.0 | 0.9146 ± 0.0044 | 0.9762 ± 0.0016 | 0.8836 ± 0.0072 | 60.0 ± 0.0 | 2.46 ± 0.00 | 8.29 ± 0.92 |
| fedprox_mu=0.01 | 0.9133 ± 0.0047 | 0.9751 ± 0.0020 | 0.8817 ± 0.0073 | 60.0 ± 0.0 | 2.46 ± 0.00 | 8.47 ± 0.68 |
| fedprox_mu=0.1 | 0.8956 ± 0.0048 | 0.9637 ± 0.0026 | 0.8540 ± 0.0078 | 60.0 ± 0.0 | 2.46 ± 0.00 | 8.58 ± 0.59 |

## Table 8. Per-seed results and 95% t-based confidence intervals (n=5 seeds)
| seed | accuracy | roc_auc | macro_f1 | mcc | rounds_run | convergence_round |
|---|---|---|---|---|---|---|
| 0 | 0.9084 | 0.9749 | 0.8733 | 0.7584 | 60.0 | 58.0 |
| 1 | 0.9183 | 0.9777 | 0.8884 | 0.7852 | 60.0 | 58.0 |
| 2 | 0.9186 | 0.9778 | 0.8912 | 0.7864 | 60.0 | 58.0 |
| 3 | 0.9159 | 0.9764 | 0.8858 | 0.7787 | 60.0 | 60.0 |
| 4 | 0.9116 | 0.9744 | 0.8793 | 0.767 | 60.0 | 60.0 |
| mean | 0.9146 | 0.9762 | 0.8836 | 0.7751 |  |  |
| SD (ddof=1) | 0.0044 | 0.0016 | 0.0072 | 0.0121 |  |  |
| 95% CI lo | 0.9091 | 0.9743 | 0.8746 | 0.7601 |  |  |
| 95% CI hi | 0.9201 | 0.9782 | 0.8926 | 0.7902 |  |  |

# Consistency checks
- bank: non-finite metric cells -> 0
- bank: Table2 accuracy mean equals raw mean -> True
- bank: Table2 accuracy SD equals np.std(ddof=1) -> True
- bank: confusion counts sum to test size per run -> True
- bank: n successful runs per experiment -> {'main': 5, 'baseline': 5, 'noniid': 5, 'scalability': 5, 'ablation': 5}
- bank_nodur: non-finite metric cells -> 0
- bank_nodur: Table2 accuracy mean equals raw mean -> True
- bank_nodur: Table2 accuracy SD equals np.std(ddof=1) -> True
- bank_nodur: confusion counts sum to test size per run -> True
- bank_nodur: n successful runs per experiment -> {'main': 5, 'baseline': 5, 'noniid': 5, 'scalability': 5, 'ablation': 5}
- hospital: non-finite metric cells -> 0
- hospital: Table2 accuracy mean equals raw mean -> True
- hospital: Table2 accuracy SD equals np.std(ddof=1) -> True
- hospital: confusion counts sum to test size per run -> True
- hospital: n successful runs per experiment -> {'main': 5, 'baseline': 5, 'noniid': 5, 'scalability': 5, 'ablation': 5}
- iot: non-finite metric cells -> 0
- iot: Table2 accuracy mean equals raw mean -> True
- iot: Table2 accuracy SD equals np.std(ddof=1) -> True
- iot: confusion counts sum to test size per run -> True
- iot: n successful runs per experiment -> {'main': 5, 'baseline': 5, 'noniid': 5, 'scalability': 5, 'ablation': 5}