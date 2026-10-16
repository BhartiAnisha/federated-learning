import numpy as np
from src.models import make_optimizer


def sample_weights(y, mode):
    """Per-sample loss weights computed from the client's OWN labels only (nothing is shared)."""
    sw = np.ones(len(y))
    if mode == "balanced":
        n, npos = len(y), int(y.sum())
        if 0 < npos < n:
            sw = np.where(y == 1, n / (2.0 * npos), n / (2.0 * (n - npos)))
    return sw


class Client:
    """A simulated client. Holds its raw records privately; only model weights leave via train()."""

    def __init__(self, cid, X, y):
        self.cid, self.X, self.y = cid, X, y
        self.n = len(y)

    def train(self, model, global_flat, cfg, lr, rng):
        model.set_flat(global_flat)
        center = [p.copy() for p in model.params()] if cfg["mu"] > 0 else None
        opt = make_optimizer(cfg["optimizer"], model.params(), lr)
        sw_all = sample_weights(self.y, cfg["class_weight"])
        bs = min(cfg["batch_size"], self.n)
        last_loss = 0.0
        for _ in range(cfg["local_epochs"]):
            perm = rng.permutation(self.n)
            tot = 0.0
            for s in range(0, self.n, bs):
                idx = perm[s:s + bs]
                loss, grads = model.loss_grads(self.X[idx], self.y[idx], sw_all[idx], rng,
                                               wd=cfg["weight_decay"], center=center, mu=cfg["mu"])
                opt.step(model.params(), grads)
                tot += loss * len(idx)
            last_loss = tot / self.n
        return model.get_flat(), self.n, last_loss
