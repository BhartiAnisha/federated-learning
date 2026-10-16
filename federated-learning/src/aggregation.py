import numpy as np


def fedavg(updates):
    """FedAvg: w_{t+1} = sum_k (n_k / N) * w_{t+1}^{(k)}, N = sum of n_k over participating clients.
    updates: list of (flat_weights, n_k)."""
    N = float(sum(n for _, n in updates))
    out = np.zeros_like(updates[0][0], dtype=np.float64)
    for w, n in updates:
        out += (n / N) * w
    return out
