"""Central server, FL training loop, and the centralized / local-only reference trainers."""
import time
import tracemalloc
import numpy as np

from src.aggregation import fedavg
from src.client import Client, sample_weights
from src.evaluation import quick_val
from src.models import MLP, make_optimizer
from src.utils import to_bytes, from_bytes


def run_federated(split, parts, cfg, seed, ckpt_path=None, track_mem=False):
    """Run FedAvg (mu=0) or FedProx (mu>0). The server only ever sees serialised weight vectors.
    Model selection / early stopping use the VALIDATION set; the test set is never touched here."""
    Xtr, ytr, Xva, yva = split["Xtr"], split["ytr"], split["Xva"], split["yva"]
    rng = np.random.default_rng(seed + 1000)
    clients = [Client(k, Xtr[p], ytr[p]) for k, p in enumerate(parts)]
    K = len(clients)
    gmodel = MLP(split["n_features"], cfg["hidden"], cfg["dropout"], seed=seed)   # server's global model
    wmodel = MLP(split["n_features"], cfg["hidden"], cfg["dropout"], seed=seed)   # scratch model for client compute
    g = gmodel.get_flat()
    m = max(1, int(round(cfg["participation"] * K)))
    if track_mem:
        tracemalloc.start()

    hist, best, bad = [], dict(loss=np.inf, round=0, flat=g.copy()), 0
    up_b = down_b = 0
    local_times, agg_times, round_times, succ = [], [], [], []
    t_all = time.perf_counter()
    for r in range(1, cfg["rounds"] + 1):
        t0 = time.perf_counter()
        chosen = rng.choice(K, size=m, replace=False)
        lr = cfg["lr"] * (cfg["lr_decay"] ** (r - 1))
        payload = to_bytes(g)                                   # download: server -> client
        updates, losses = [], []
        for k in chosen:
            down_b += len(payload)
            tl = time.perf_counter()
            flat, n, loss = clients[k].train(wmodel, from_bytes(payload), cfg, lr, rng)
            local_times.append(time.perf_counter() - tl)
            msg = to_bytes(flat)                                # upload: client -> server
            up_b += len(msg)
            updates.append((from_bytes(msg), n)); losses.append((loss, n))
        ta = time.perf_counter()
        g = fedavg(updates)
        agg_times.append(time.perf_counter() - ta)
        succ.append(len(updates))
        gmodel.set_flat(g)
        vl, va, vauc = quick_val(yva, gmodel.predict_proba(Xva))
        trl = sum(l * n for l, n in losses) / sum(n for _, n in losses)
        hist.append(dict(round=r, train_loss=trl, val_loss=vl, val_acc=va, val_auc=vauc))
        round_times.append(time.perf_counter() - t0)
        if vl < best["loss"] - 1e-5:
            best, bad = dict(loss=vl, round=r, flat=g.copy()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    total = time.perf_counter() - t_all
    peak = None
    if track_mem:
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
    if ckpt_path:
        np.savez(ckpt_path, weights=best["flat"], best_round=best["round"])
    gmodel.set_flat(best["flat"])
    size_mb = gmodel.n_params * 4 / 1e6
    return dict(model=gmodel, history=hist, best_round=best["round"], rounds_run=len(hist), train_time=total,
                avg_local_time=float(np.mean(local_times)), avg_agg_time=float(np.mean(agg_times)),
                avg_round_time=float(np.mean(round_times)), upload_mb=up_b / 1e6, download_mb=down_b / 1e6,
                total_comm_mb=(up_b + down_b) / 1e6, n_params=gmodel.n_params, model_mb=size_mb,
                clients_per_round=m, mean_success_per_round=float(np.mean(succ)), peak_mem_mb=peak,
                parts_sizes=[len(p) for p in parts])


def client_level_accuracy(model, split, parts):
    """Accuracy / AUC of the final global model on each client's own (private) training records."""
    from sklearn.metrics import roc_auc_score
    accs, aucs = [], []
    for p in parts:
        pr = model.predict_proba(split["Xtr"][p])
        accs.append(float(np.mean((pr >= 0.5) == split["ytr"][p])))
        if len(np.unique(split["ytr"][p])) == 2:
            aucs.append(float(roc_auc_score(split["ytr"][p], pr)))
    return accs, aucs


def train_centralized(X, y, Xva, yva, cfg, seed, max_epochs=100, patience=15):
    """Non-federated reference (all training records pooled). Same MLP, early stopping on validation loss."""
    rng = np.random.default_rng(seed + 2000)
    model = MLP(X.shape[1], cfg["hidden"], cfg["dropout"], seed=seed)
    if len(np.unique(y)) < 2:                              # degenerate client: predict the prior
        model.b[-1][...] = np.log((y.mean() + 1e-3) / (1 - y.mean() + 1e-3))
        for W in model.W:
            W[...] = 0.0
        return model, 0
    opt = make_optimizer(cfg["optimizer"], model.params(), cfg["lr"])
    sw_all = sample_weights(y, cfg["class_weight"])
    bs = min(cfg["batch_size"], len(y))
    best, best_flat, bad, best_ep = np.inf, model.get_flat(), 0, 0
    for ep in range(1, max_epochs + 1):
        perm = rng.permutation(len(y))
        for s in range(0, len(y), bs):
            idx = perm[s:s + bs]
            _, grads = model.loss_grads(X[idx], y[idx], sw_all[idx], rng, wd=cfg["weight_decay"])
            opt.step(model.params(), grads)
        vl, _, _ = quick_val(yva, model.predict_proba(Xva))
        if vl < best - 1e-5:
            best, best_flat, bad, best_ep = vl, model.get_flat(), 0, ep
        else:
            bad += 1
            if bad >= patience:
                break
    model.set_flat(best_flat)
    return model, best_ep
