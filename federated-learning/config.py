"""Central configuration for the federated learning project."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")
CKPT_DIR = os.path.join(ROOT, "checkpoints")

SEEDS = [0, 1, 2, 3, 4]
SPLIT = dict(test=0.15, val=0.15)          # remaining 70 % is the federated training pool
THRESHOLD = 0.5                            # default decision threshold (not tuned)

# Default FL hyper-parameters (starting point; tuned on validation data only)
DEFAULTS = dict(
    n_clients=10, participation=1.0, rounds=80, patience=10,
    local_epochs=3, batch_size=32, optimizer="adam", lr=0.01, lr_decay=1.0,
    hidden=(32,), dropout=0.0, weight_decay=0.0, class_weight=None,
    mu=0.0,                                  # FedProx proximal coefficient (0 => FedAvg)
)

# Per-dataset overrides
DATASETS = {
    "hospital":   dict(file="Hospital_management.csv", label="Healthcare (Breast Cancer Wisconsin, Original)"),
    "bank":       dict(file="banking_management.csv", label="Finance (UCI Bank Marketing)"),
    "bank_nodur":  dict(file="banking_management.csv", label="Finance, call duration removed (leakage-safe variant)"),
    "iot":        dict(file="IOT-temp.csv", label="IoT (temperature log, In/Out)",
                       overrides=dict(batch_size=256, rounds=60, patience=8)),
}

# Non-IID settings: Dirichlet concentration (smaller = more skewed)
NONIID = {"iid": None, "moderate (Dir 0.5)": 0.5, "strong (Dir 0.1)": 0.1}
SCALE_CLIENTS = [5, 10, 20, 50]
FEDPROX_MU = 0.01          # fixed, not tuned

# Random-search space for validation-only tuning
SEARCH_SPACE = dict(
    hidden=[(16,), (32,), (64,), (64, 32)],
    dropout=[0.0, 0.1, 0.3],
    weight_decay=[0.0, 1e-4, 1e-3],
    local_epochs=[1, 3, 5],
    batch_size=[16, 32, 64],
    lr_decay=[1.0, 0.97],
    class_weight=[None, "balanced"],
    optimizer=["adam", "sgd"],
    lr_adam=[0.001, 0.003, 0.01],
    lr_sgd=[0.03, 0.1, 0.3],
)
N_TUNE = 30
