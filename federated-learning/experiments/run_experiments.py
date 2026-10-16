"""End-to-end experiment driver: tuning (validation only) -> main -> baselines -> non-IID -> scalability -> ablation.

Usage:  python main.py --datasets hospital bank --stages all
"""
import argparse, copy, json, os, sys, time, traceback
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config as C
from src.data_preprocessing import (load_dataset, split_and_preprocess, make_partition, describe_partition)
from src.evaluation import compute_metrics
from src.server import run_federated, client_level_accuracy, train_centralized
from src.utils import set_seed, env_info, file_md5, save_json

STAGES = ["tune", "main", "baselines", "noniid", "scalability", "ablation"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def base_cfg(name):
    cfg = copy.deepcopy(C.DEFAULTS)
    cfg.update(C.DATASETS[name].get("overrides", {}))
    return cfg


def tuned_path(name):
    return os.path.join(C.RESULTS_DIR, f"best_config_{name}.json")


def load_cfg(name):
    cfg = base_cfg(name)
    if os.path.exists(tuned_path(name)):
        with open(tuned_path(name)) as f:
            t = json.load(f)
        t["hidden"] = tuple(t["hidden"])
        cfg.update(t)
    return cfg


def fl_row(split, parts, cfg, seed, extra, ckpt=None, track_mem=False, keep=False):
    res = run_federated(split, parts, cfg, seed, ckpt_path=ckpt, track_mem=track_mem)
    model = res["model"]
    p = model.predict_proba(split["Xte"])
    m = compute_metrics(split["yte"], p, C.THRESHOLD)         # test set touched once, after model selection
    accs, aucs = client_level_accuracy(model, split, parts)
    row = dict(extra, seed=seed, **m, threshold=C.THRESHOLD, threshold_source="default 0.5",
               train_time=res["train_time"], avg_local_time=res["avg_local_time"],
               avg_agg_time=res["avg_agg_time"], avg_round_time=res["avg_round_time"],
               upload_mb=res["upload_mb"], download_mb=res["download_mb"], total_comm_mb=res["total_comm_mb"],
               n_params=res["n_params"], model_mb=res["model_mb"], peak_mem_mb=res["peak_mem_mb"],
               participation=cfg["participation"], clients_per_round=res["clients_per_round"],
               success_per_round=res["mean_success_per_round"], rounds_run=res["rounds_run"],
               convergence_round=res["best_round"],
               client_acc_mean=float(np.mean(accs)), client_acc_std=float(np.std(accs, ddof=1)) if len(accs) > 1 else np.nan,
               client_acc_min=float(np.min(accs)),
               client_auc_mean=float(np.mean(aucs)) if aucs else np.nan,
               client_auc_std=float(np.std(aucs, ddof=1)) if len(aucs) > 1 else np.nan)
    if keep:
        return row, res, p
    return row


def get_split(X, y, seed):
    return split_and_preprocess(X, y, seed)


# --------------------------------------------------------------------------------------------- tuning
def stage_tune(name, X, y, quick):
    cfg0 = base_cfg(name)
    split = get_split(X, y, 0)
    parts = make_partition(split["ytr"], cfg0["n_clients"], None, 0)
    rng = np.random.default_rng(123)
    S = C.SEARCH_SPACE
    cands = [dict()]                                          # first candidate: untouched defaults
    for _ in range(4 if quick else C.N_TUNE):
        c = {k: S[k][rng.integers(len(S[k]))] for k in
             ["hidden", "dropout", "weight_decay", "local_epochs", "batch_size", "lr_decay", "class_weight", "optimizer"]}
        c["lr"] = S["lr_adam"][rng.integers(3)] if c["optimizer"] == "adam" else S["lr_sgd"][rng.integers(3)]
        if name == "iot":
            c["batch_size"] = int(rng.choice([128, 256, 512]))
        cands.append(c)
    rows = []
    for i, c in enumerate(cands):
        cfg = {**cfg0, **c}
        if quick:
            cfg["rounds"] = 15
        t0 = time.time()
        res = run_federated(split, parts, cfg, 0)
        h = res["history"][res["best_round"] - 1]
        score = h["val_acc"] + h["val_auc"]                    # validation-only selection criterion
        rows.append(dict(idx=i, score=score, val_acc=h["val_acc"], val_auc=h["val_auc"], val_loss=h["val_loss"],
                         best_round=res["best_round"], secs=time.time() - t0, **{k: str(v) for k, v in cfg.items()}))
        log(f"[{name}] tune {i+1}/{len(cands)} val_acc={h['val_acc']:.4f} val_auc={h['val_auc']:.4f}")
    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    df.to_csv(os.path.join(C.RESULTS_DIR, f"tuning_{name}.csv"), index=False)
    best = cands[int(df.iloc[0]["idx"])]
    final = {**cfg0, **best}
    save_json(final, tuned_path(name))
    log(f"[{name}] best config: {final}")


# --------------------------------------------------------------------------------------------- main
def stage_main(name, X, y, meta, seeds, quick):
    cfg = load_cfg(name)
    if quick:
        cfg["rounds"] = 20
    rows, curves = [], {}
    for s in seeds:
        set_seed(s)
        split = get_split(X, y, s)
        parts = make_partition(split["ytr"], cfg["n_clients"], None, s)
        ck = os.path.join(C.CKPT_DIR, f"{name}_fedavg_seed{s}.npz")
        row, res, p = fl_row(split, parts, cfg, s, dict(dataset=name, experiment="main", setting="FedAvg (IID, K=%d)" % cfg["n_clients"]),
                             ckpt=ck, keep=True)
        rows.append(row)
        curves[str(s)] = res["history"]
        if s == seeds[0]:
            np.savez(os.path.join(C.RESULTS_DIR, f"test_probs_{name}.npz"), y=split["yte"], p=p)
        log(f"[{name}] main seed {s}: acc={row['accuracy']:.4f} auc={row['roc_auc']:.4f} rounds={row['rounds_run']}")
    save_json(curves, os.path.join(C.RESULTS_DIR, f"curves_{name}.json"))
    return rows


# --------------------------------------------------------------------------------------------- baselines
def stage_baselines(name, X, y, seeds, quick):
    cfg = load_cfg(name)
    if quick:
        cfg["rounds"] = 20
    max_ep = 30 if name == "iot" else 100
    rows = []
    for s in seeds:
        set_seed(s)
        split = get_split(X, y, s)
        parts = make_partition(split["ytr"], cfg["n_clients"], None, s)
        yte, Xte = split["yte"], split["Xte"]

        def add(setting, m, **kw):
            rows.append(dict(dataset=name, experiment="baseline", setting=setting, seed=s, **m, **kw))

        t = time.perf_counter()
        mdl, ep = train_centralized(split["Xtr"], split["ytr"], split["Xva"], split["yva"], cfg, s, max_ep)
        add("Centralized MLP (non-FL)", compute_metrics(yte, mdl.predict_proba(Xte)), train_time=time.perf_counter() - t)

        t = time.perf_counter()
        lcfg = {**cfg, "hidden": (), "dropout": 0.0}
        mdl, ep = train_centralized(split["Xtr"], split["ytr"], split["Xva"], split["yva"], lcfg, s, max_ep)
        add("Centralized Logistic Regression (non-FL)", compute_metrics(yte, mdl.predict_proba(Xte)), train_time=time.perf_counter() - t)

        t = time.perf_counter()
        lr = LogisticRegression(max_iter=2000).fit(split["Xtr"], split["ytr"])
        add("Centralized sklearn LogReg (non-FL)", compute_metrics(yte, lr.predict_proba(Xte)[:, 1]), train_time=time.perf_counter() - t)

        t = time.perf_counter()
        rf = RandomForestClassifier(n_estimators=200, random_state=s, n_jobs=1).fit(split["Xtr"], split["ytr"])
        add("Centralized Random Forest (non-FL)", compute_metrics(yte, rf.predict_proba(Xte)[:, 1]), train_time=time.perf_counter() - t)

        # local-only: every client trains on its own records only (validation set used for early stopping)
        t = time.perf_counter()
        ms = []
        for p in parts:
            m_, _ = train_centralized(split["Xtr"][p], split["ytr"][p], split["Xva"], split["yva"], cfg, s, max_ep // 2)
            ms.append(compute_metrics(yte, m_.predict_proba(Xte)))
        avg = {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]}
        add("Local-only (mean over clients)", avg, train_time=time.perf_counter() - t,
            client_acc_std=float(np.std([m["accuracy"] for m in ms], ddof=1)),
            client_acc_min=float(np.min([m["accuracy"] for m in ms])),
            client_acc_mean=float(np.mean([m["accuracy"] for m in ms])))

        # FedProx (mu fixed, not tuned)
        pcfg = {**cfg, "mu": C.FEDPROX_MU}
        rows.append(fl_row(split, parts, pcfg, s, dict(dataset=name, experiment="baseline", setting=f"FedProx (mu={C.FEDPROX_MU}, IID)")))
        log(f"[{name}] baselines seed {s} done")
    return rows


# --------------------------------------------------------------------------------------------- non-IID
def stage_noniid(name, X, y, seeds, quick):
    cfg = load_cfg(name)
    if quick:
        cfg["rounds"] = 20
    rows = []
    for s in seeds:
        set_seed(s)
        split = get_split(X, y, s)
        for label, alpha in C.NONIID.items():
            parts = make_partition(split["ytr"], cfg["n_clients"], alpha, s)
            d = describe_partition(parts, split["ytr"])
            row = fl_row(split, parts, cfg, s, dict(dataset=name, experiment="noniid", setting=label,
                                                    alpha=alpha if alpha is not None else np.inf,
                                                    part_size_min=d["size_min"], part_size_max=d["size_max"],
                                                    part_pos_rate_min=d["pos_rate_min"], part_pos_rate_max=d["pos_rate_max"]))
            rows.append(row)
        log(f"[{name}] noniid seed {s} done")
    return rows


# --------------------------------------------------------------------------------------------- scalability
def stage_scalability(name, X, y, seeds, quick):
    cfg = load_cfg(name)
    if quick:
        cfg["rounds"] = 20
    ks = [5, 10, 20] if quick else C.SCALE_CLIENTS
    rows = []
    for s in seeds:
        set_seed(s)
        split = get_split(X, y, s)
        for K in ks:
            parts = make_partition(split["ytr"], K, None, s)      # same pooled data, split K ways
            kcfg = {**cfg, "n_clients": K}
            rows.append(fl_row(split, parts, kcfg, s, dict(dataset=name, experiment="scalability", setting=f"K={K}", n_clients=K,
                                                           train_records=len(split["ytr"])), track_mem=True))
        log(f"[{name}] scalability seed {s} done")
    return rows


# --------------------------------------------------------------------------------------------- ablation
def stage_ablation(name, X, y, seeds, quick):
    cfg = load_cfg(name)
    if quick:
        cfg["rounds"] = 20
    variants = {}
    for e in (1, 3, 5):
        variants[f"local_epochs={e}"] = dict(local_epochs=e)
    for f in (0.3, 1.0, 3.0):
        variants[f"lr={cfg['lr'] * f:g}"] = dict(lr=cfg["lr"] * f)
    for pr in (0.2, 0.5, 1.0):
        variants[f"participation={pr}"] = dict(participation=pr)
    other = "sgd" if cfg["optimizer"] == "adam" else "adam"
    variants[f"optimizer={cfg['optimizer']}"] = dict(optimizer=cfg["optimizer"])
    variants[f"optimizer={other}"] = dict(optimizer=other, lr=(0.1 if other == "sgd" else 0.01))
    for mu in (0.0, 0.01, 0.1):
        variants[f"fedprox_mu={mu}"] = dict(mu=mu)
    rows = []
    for s in seeds:
        set_seed(s)
        split = get_split(X, y, s)
        parts = make_partition(split["ytr"], cfg["n_clients"], None, s)
        for label, ov in variants.items():
            rows.append(fl_row(split, parts, {**cfg, **ov}, s, dict(dataset=name, experiment="ablation", setting=label)))
        log(f"[{name}] ablation seed {s} done")
    return rows


# --------------------------------------------------------------------------------------------- orchestration
def table1_info(name, X, y, meta, seed=0):
    split = get_split(X, y, seed)
    info = dict(meta)
    info.update(final_features_after_encoding=split["n_features"], n_train=len(split["ytr"]), n_val=len(split["yva"]),
                n_test=len(split["yte"]), train_pos_rate=float(split["ytr"].mean()), val_pos_rate=float(split["yva"].mean()),
                test_pos_rate=float(split["yte"].mean()), majority_baseline_test_acc=float(max(split["yte"].mean(), 1 - split["yte"].mean())))
    info["partitions"] = {}
    for label, alpha in C.NONIID.items():
        parts = make_partition(split["ytr"], 10, alpha, seed)
        info["partitions"][label] = describe_partition(parts, split["ytr"])
    info["md5"] = file_md5(os.path.join(C.DATA_DIR, C.DATASETS[name]["file"]))
    return info


def run_dataset(name, stages, seeds, quick, drop_duration=False):
    os.makedirs(C.RESULTS_DIR, exist_ok=True); os.makedirs(C.CKPT_DIR, exist_ok=True)
    X, y, meta = load_dataset(name.replace("_nodur", ""), drop_duration=drop_duration)
    log(f"[{name}] loaded {meta['original_shape']} -> {X.shape}, classes={meta['classes']}, dup removed={meta['duplicates_removed']}")
    info = table1_info(name.replace("_nodur", ""), X, y, meta)
    save_json(info, os.path.join(C.RESULTS_DIR, f"dataset_info_{name}.json"))
    out_csv = os.path.join(C.RESULTS_DIR, f"raw_{name}.csv")
    existing = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()
    allrows = []
    fns = dict(main=lambda: stage_main(name, X, y, meta, seeds, quick), baselines=lambda: stage_baselines(name, X, y, seeds, quick),
               noniid=lambda: stage_noniid(name, X, y, seeds, quick), scalability=lambda: stage_scalability(name, X, y, seeds, quick),
               ablation=lambda: stage_ablation(name, X, y, seeds, quick))
    for st in stages:
        t0 = time.time()
        try:
            if st == "tune":
                stage_tune(name, X, y, quick)
                continue
            rows = fns[st]()
            for r in rows:
                r["dataset"] = name
            exp = rows[0]["experiment"]
            if len(existing):
                existing = existing[existing["experiment"] != exp]       # replace that experiment's old rows
            existing = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
            existing.to_csv(out_csv, index=False)
            log(f"[{name}] stage {st} finished in {time.time()-t0:.0f}s")
        except Exception:
            log(f"[{name}] stage {st} FAILED"); traceback.print_exc()
    save_json(dict(env=env_info(), seeds=seeds, split=C.SPLIT, threshold=C.THRESHOLD, quick=quick,
                   config=load_cfg(name), fedprox_mu=C.FEDPROX_MU, noniid=C.NONIID),
              os.path.join(C.RESULTS_DIR, f"run_info_{name}.json"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["hospital", "bank"])
    ap.add_argument("--stages", nargs="+", default=["all"])
    ap.add_argument("--seeds", type=int, default=len(C.SEEDS))
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 seeds, few rounds, tiny search")
    ap.add_argument("--results-dir", default=None)
    a = ap.parse_args(argv)
    if a.results_dir:
        C.RESULTS_DIR = a.results_dir
    stages = STAGES if "all" in a.stages else a.stages
    seeds = C.SEEDS[:2] if a.quick else C.SEEDS[:a.seeds]
    for d in a.datasets:
        run_dataset(d, stages, seeds, a.quick, drop_duration=d.endswith("_nodur"))


if __name__ == "__main__":
    main()
