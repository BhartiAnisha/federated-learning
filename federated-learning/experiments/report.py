"""Build Tables 1-8, figures (PNG+PDF), consistency checks and a Results section strictly from results/raw_*.csv."""
import glob, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from sklearn.metrics import roc_curve, precision_recall_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config as C
from src.utils import summarize, paired_test, fmt, save_json

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 100})
MAIN_METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "macro_f1", "weighted_f1", "roc_auc", "pr_auc",
                "balanced_accuracy", "mcc", "fpr", "fnr", "log_loss", "brier"]


def savefig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(C.FIG_DIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def agg(df, cols, nd=4):
    out = {}
    for c in cols:
        if c in df and df[c].notna().any():
            s = summarize(df[c].values)
            out[c] = fmt(s["mean"], s["std"], nd)
        else:
            out[c] = "n/a"
    return out


def load(name):
    p = os.path.join(C.RESULTS_DIR, f"raw_{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def dataset_names():
    return [os.path.basename(p)[4:-4] for p in sorted(glob.glob(os.path.join(C.RESULTS_DIR, "raw_*.csv")))]


def main():
    os.makedirs(C.FIG_DIR, exist_ok=True)
    tdir = os.path.join(C.RESULTS_DIR, "tables"); os.makedirs(tdir, exist_ok=True)
    names = dataset_names()
    all_md, results_json, checks = [], {}, []
    summary_rows, seed_rows, scal_rows, cm_rows = [], [], [], []

    for name in names:
        df = load(name)
        info = json.load(open(os.path.join(C.RESULTS_DIR, f"dataset_info_{name}.json")))
        cfg = json.load(open(os.path.join(C.RESULTS_DIR, f"best_config_{name}.json")))
        md = [f"\n# Dataset: {name}\n"]

        # ---- finiteness check
        num = df.select_dtypes(include=[float, int])
        bad = num[MAIN_METRICS].replace([np.inf, -np.inf], np.nan).isna().sum() if set(MAIN_METRICS) <= set(num.columns) else None
        checks.append((name, "non-finite metric cells", int(bad.sum()) if bad is not None else "n/a"))

        # ---- Table 1
        t1 = pd.DataFrame([
            ["Original shape (rows x cols)", str(info["original_shape"])],
            ["Exact duplicate records removed", info["duplicates_removed"]],
            ["Rows after cleaning", info["final_rows_before_encoding"]],
            ["Raw features / features after encoding", f"{info['n_raw_features']} / {info['final_features_after_encoding']}"],
            ["Classes (count)", info["classes"]],
            ["Missing values (original file)", info["missing_values"]],
            ["Train / Val / Test", f"{info['n_train']} / {info['n_val']} / {info['n_test']}"],
            ["Positive rate train/val/test", f"{info['train_pos_rate']:.3f} / {info['val_pos_rate']:.3f} / {info['test_pos_rate']:.3f}"],
            ["Majority-class test accuracy (trivial baseline)", f"{info['majority_baseline_test_acc']:.4f}"],
        ] + [[f"Client sizes, K=10, {k}", f"{v['size_min']}–{v['size_max']} (mean {v['size_mean']:.1f}); client positive rate {v['pos_rate_min']:.2f}–{v['pos_rate_max']:.2f}"]
             for k, v in info["partitions"].items()], columns=["Item", "Value"])
        t1.to_csv(os.path.join(tdir, f"table1_{name}.csv"), index=False)
        md += ["## Table 1. Dataset characteristics and client distribution (seed-0 split)", md_table(t1)]

        main_df = df[df.experiment == "main"]
        base = df[df.experiment == "baseline"]
        # ---- Table 2
        t2 = pd.DataFrame([dict(Metric=m, **{"Mean ± SD (n=%d)" % len(main_df): agg(main_df, [m])[m],
                                              "Min": f"{main_df[m].min():.4f}", "Max": f"{main_df[m].max():.4f}",
                                              "95% CI": "[%.4f, %.4f]" % (summarize(main_df[m].values)["ci_lo"], summarize(main_df[m].values)["ci_hi"])})
                           for m in MAIN_METRICS])
        t2.to_csv(os.path.join(tdir, f"table2_{name}.csv"), index=False)
        md += ["\n## Table 2. FedAvg (IID, K=%d) test performance, threshold=0.5 (default, not tuned), mean ± SD across %d seeds" % (cfg["n_clients"], len(main_df)), md_table(t2)]
        for m in MAIN_METRICS:
            s = summarize(main_df[m].values)
            summary_rows.append(dict(dataset=name, method="FedAvg IID K=10", metric=m, **s))
        for _, r in main_df.iterrows():
            seed_rows.append({k: r[k] for k in ["dataset", "seed"] + MAIN_METRICS + ["rounds_run", "convergence_round", "train_time", "total_comm_mb"]})
        cm = main_df[["tn", "fp", "fn", "tp"]].sum()
        cm_rows.append(dict(dataset=name, **{k: int(v) for k, v in cm.items()}, note="summed over seeds"))

        # ---- Table 3
        methods = [("FedAvg (IID, K=%d)" % cfg["n_clients"], main_df)] + [(s, base[base.setting == s]) for s in base.setting.unique()]
        n_ = {s: d.sort_values("seed") for s, d in methods}
        ref = n_[methods[0][0]]
        rows = []
        for s, d in methods:
            d = d.sort_values("seed")
            r = dict(Method=s, **agg(d, ["accuracy", "recall", "macro_f1", "roc_auc", "mcc"]))
            if s != methods[0][0] and len(d) == len(ref) and len(d) > 1:
                ta = paired_test(ref["accuracy"].values, d["accuracy"].values)
                tc = paired_test(ref["roc_auc"].values, d["roc_auc"].values)
                r["Δacc vs FedAvg (FedAvg−X)"] = f"{ta['mean_diff']:+.4f}"
                r["paired t p (acc)"] = f"{ta['p']:.3f}" if np.isfinite(ta["p"]) else "n/a"
                r["d_z (acc)"] = f"{ta['dz']:.2f}" if np.isfinite(ta["dz"]) else "n/a"
                r["paired t p (AUC)"] = f"{tc['p']:.3f}" if np.isfinite(tc["p"]) else "n/a"
            rows.append(r)
        t3 = pd.DataFrame(rows).fillna("")
        t3.to_csv(os.path.join(tdir, f"table3_{name}.csv"), index=False)
        md += ["\n## Table 3. Comparison with centralized, local-only and FL baselines (centralized rows are NOT federated; paired tests on the same seeds/splits, no multiple-comparison correction)", md_table(t3)]

        # ---- Table 4
        ni = df[df.experiment == "noniid"]
        rows = []
        for s in ni.setting.unique():
            d = ni[ni.setting == s]
            rows.append(dict(Setting=s, **agg(d, ["accuracy", "macro_f1", "recall", "roc_auc"]),
                             **{"Train time (s)": agg(d, ["train_time"], 2)["train_time"], "Comm (MB)": agg(d, ["total_comm_mb"], 2)["total_comm_mb"],
                                "Rounds": agg(d, ["rounds_run"], 1)["rounds_run"],
                                "Client-level acc mean (SD across clients)": "%.4f (%.4f)" % (d.client_acc_mean.mean(), d.client_acc_std.mean())}))
        t4 = pd.DataFrame(rows)
        t4.to_csv(os.path.join(tdir, f"table4_{name}.csv"), index=False)
        md += ["\n## Table 4. IID vs non-IID (Dirichlet label skew, K=10). Client-level accuracy = global model on each client's own training records, averaged over seeds; it is separate from across-seed SD", md_table(t4)]

        # ---- Tables 5/6
        sc = df[df.experiment == "scalability"]
        rows5, rows6 = [], []
        for K in sorted(sc.n_clients.unique()):
            d = sc[sc.n_clients == K]
            rows5.append(dict(Clients=int(K), **agg(d, ["accuracy", "roc_auc", "macro_f1"]), Rounds=agg(d, ["rounds_run"], 1)["rounds_run"],
                              **{"Convergence round": agg(d, ["convergence_round"], 1)["convergence_round"],
                                 "Records/client": "%.1f" % (d.train_records.iloc[0] / K)}))
            rows6.append(dict(Clients=int(K), **{"Total time (s)": agg(d, ["train_time"], 2)["train_time"], "Avg local (s)": agg(d, ["avg_local_time"], 4)["avg_local_time"],
                                                 "Avg aggregation (ms)": "%.3f ± %.3f" % (d.avg_agg_time.mean() * 1e3, d.avg_agg_time.std(ddof=1) * 1e3),
                                                 "Time/round (s)": agg(d, ["avg_round_time"], 3)["avg_round_time"],
                                                 "Upload (MB)": agg(d, ["upload_mb"], 3)["upload_mb"], "Download (MB)": agg(d, ["download_mb"], 3)["download_mb"],
                                                 "Total comm (MB)": agg(d, ["total_comm_mb"], 3)["total_comm_mb"],
                                                 "Params": int(d.n_params.iloc[0]), "Model (MB)": "%.5f" % d.model_mb.iloc[0],
                                                 "Peak Py/NumPy mem (MB)": agg(d, ["peak_mem_mb"], 2)["peak_mem_mb"],
                                                 "Clients/round": "%.1f" % d.clients_per_round.mean(), "Successful/round": "%.1f" % d.success_per_round.mean()}))
            for _, r in d.iterrows():
                scal_rows.append({k: r[k] for k in ["dataset", "seed", "n_clients", "accuracy", "roc_auc", "train_time", "avg_local_time", "avg_agg_time", "avg_round_time",
                                                    "upload_mb", "download_mb", "total_comm_mb", "n_params", "model_mb", "peak_mem_mb", "rounds_run", "convergence_round"]})
        t5, t6 = pd.DataFrame(rows5), pd.DataFrame(rows6)
        t5.to_csv(os.path.join(tdir, f"table5_{name}.csv"), index=False); t6.to_csv(os.path.join(tdir, f"table6_{name}.csv"), index=False)
        md += ["\n## Table 5. Scalability (same pooled training data split across K clients, IID, early stopping on)", md_table(t5),
               "\n## Table 6. Communication, runtime, memory, model size. Communication = float32 weight payloads actually serialised (4 B/param) x participating clients x rounds, both directions; excludes protocol headers, compression, encryption and retransmissions. Memory = tracemalloc peak of Python/NumPy allocations in the run (not process RSS). Timings are single-process simulation on a shared 2-core CPU, so they are not network latencies.", md_table(t6)]

        # ---- Table 7
        ab = df[df.experiment == "ablation"]
        rows = []
        for s in ab.setting.unique():
            d = ab[ab.setting == s]
            rows.append(dict(Variant=s, **agg(d, ["accuracy", "roc_auc", "macro_f1"]), Rounds=agg(d, ["rounds_run"], 1)["rounds_run"],
                             **{"Comm (MB)": agg(d, ["total_comm_mb"], 2)["total_comm_mb"], "Time (s)": agg(d, ["train_time"], 2)["train_time"]}))
        t7 = pd.DataFrame(rows)
        t7.to_csv(os.path.join(tdir, f"table7_{name}.csv"), index=False)
        md += ["\n## Table 7. Ablations around the tuned configuration (one factor changed at a time; K=10, IID). Client-count ablation is Table 5.", md_table(t7)]

        # ---- Table 8
        t8 = main_df.sort_values("seed")[["seed", "accuracy", "roc_auc", "macro_f1", "mcc", "rounds_run", "convergence_round"]].copy()
        t8 = t8.round(4)
        ci = {m: summarize(main_df[m].values) for m in ["accuracy", "roc_auc", "macro_f1", "mcc"]}
        extra = pd.DataFrame([dict(seed="mean", **{m: round(ci[m]["mean"], 4) for m in ci}), dict(seed="SD (ddof=1)", **{m: round(ci[m]["std"], 4) for m in ci}),
                              dict(seed="95% CI lo", **{m: round(ci[m]["ci_lo"], 4) for m in ci}), dict(seed="95% CI hi", **{m: round(ci[m]["ci_hi"], 4) for m in ci})])
        t8 = pd.concat([t8, extra], ignore_index=True).fillna("")
        t8.to_csv(os.path.join(tdir, f"table8_{name}.csv"), index=False)
        md += ["\n## Table 8. Per-seed results and 95%% t-based confidence intervals (n=%d seeds)" % len(main_df), md_table(t8)]

        # ---- consistency check: summary == raw
        s_acc = summarize(main_df.accuracy.values)
        checks.append((name, "Table2 accuracy mean equals raw mean", bool(np.isclose(s_acc["mean"], main_df.accuracy.mean()))))
        checks.append((name, "Table2 accuracy SD equals np.std(ddof=1)", bool(np.isclose(s_acc["std"], np.std(main_df.accuracy.values, ddof=1)))))
        checks.append((name, "confusion counts sum to test size per run", bool(((main_df.tn + main_df.fp + main_df.fn + main_df.tp) == info["n_test"]).all())))
        checks.append((name, "n successful runs per experiment", {e: int(df[df.experiment == e].seed.nunique()) for e in df.experiment.unique()}))

        # ---------------------------------------------------------------- figures
        curves = json.load(open(os.path.join(C.RESULTS_DIR, f"curves_{name}.json")))
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
        for s, h in curves.items():
            r = [x["round"] for x in h]
            ax[0].plot(r, [x["train_loss"] for x in h], color="C0", alpha=.5, label="train (client-avg)" if s == "0" else None)
            ax[0].plot(r, [x["val_loss"] for x in h], color="C1", alpha=.5, label="validation" if s == "0" else None)
            ax[1].plot(r, [x["val_acc"] for x in h], color="C2", alpha=.6)
            ax[2].plot(r, [x["val_auc"] for x in h], color="C3", alpha=.6)
        ax[0].set(title="Loss (each line = one seed)", xlabel="Communication round", ylabel="BCE"); ax[0].legend()
        ax[1].set(title="Validation accuracy", xlabel="Communication round"); ax[2].set(title="Validation ROC-AUC", xlabel="Communication round")
        savefig(fig, f"{name}_convergence")

        tp = np.load(os.path.join(C.RESULTS_DIR, f"test_probs_{name}.npz"))
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
        fpr, tpr, _ = roc_curve(tp["y"], tp["p"]); ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "k--", lw=.8)
        ax[0].set(title=f"ROC (seed 0, AUC={main_df[main_df.seed == 0].roc_auc.iloc[0]:.3f})", xlabel="FPR", ylabel="TPR")
        pr, rc, _ = precision_recall_curve(tp["y"], tp["p"]); ax[1].plot(rc, pr)
        ax[1].set(title=f"Precision-recall (seed 0, AP={main_df[main_df.seed == 0].pr_auc.iloc[0]:.3f})", xlabel="Recall", ylabel="Precision")
        r0 = main_df[main_df.seed == 0].iloc[0]
        M = np.array([[r0.tn, r0.fp], [r0.fn, r0.tp]])
        ax[2].imshow(M, cmap="Blues"); ax[2].grid(False)
        for i in range(2):
            for j in range(2):
                ax[2].text(j, i, int(M[i, j]), ha="center", va="center", color="white" if M[i, j] > M.max() / 2 else "black", fontsize=12)
        ax[2].set(title="Confusion matrix (seed 0, thr 0.5)", xticks=[0, 1], yticks=[0, 1], xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"])
        savefig(fig, f"{name}_roc_pr_cm")

        ks = sorted(sc.n_clients.unique())
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
        for a_, col, yl in zip(ax, ["accuracy", "total_comm_mb", "train_time"], ["Test accuracy", "Total communication (MB)", "Training time (s)"]):
            mu = [sc[sc.n_clients == k][col].mean() for k in ks]; sd = [sc[sc.n_clients == k][col].std(ddof=1) for k in ks]
            a_.errorbar(ks, mu, yerr=sd, marker="o", capsize=3); a_.set(xlabel="Number of clients K", ylabel=yl, xscale="log")
            a_.set_xticks([int(k) for k in ks]); a_.set_xticklabels([str(int(k)) for k in ks]); a_.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            a_.set_title(yl + " vs K (mean ± SD)")
        savefig(fig, f"{name}_scalability")

        fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
        labs = list(ni.setting.unique())
        for a_, col in zip(ax, ["accuracy", "roc_auc"]):
            mu = [ni[ni.setting == l][col].mean() for l in labs]; sd = [ni[ni.setting == l][col].std(ddof=1) for l in labs]
            a_.bar(range(len(labs)), mu, yerr=sd, capsize=4, color=["C0", "C1", "C3"][:len(labs)])
            a_.set_xticks(range(len(labs))); a_.set_xticklabels([l.replace(" (", "\n(") for l in labs], fontsize=7)
            lo = min(mu) - 2 * np.nanmax(sd) - 0.02; a_.set_ylim(max(0, lo), 1.0); a_.set_title(col + " (mean ± SD over seeds)")
        savefig(fig, f"{name}_noniid")

        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        order = list(t3.Method); vals = [d for s, d in methods]
        mu = [d.accuracy.mean() for d in vals]; sd = [d.accuracy.std(ddof=1) for d in vals]
        ax.barh(range(len(vals)), mu, xerr=sd, capsize=3, color=["C3"] + ["C7"] * (len(vals) - 1))
        ax.set_yticks(range(len(vals))); ax.set_yticklabels([s for s, _ in methods], fontsize=7); ax.invert_yaxis()
        ax.set_xlim(max(0, min(mu) - 3 * np.nanmax(sd) - 0.03), 1.0); ax.set_title("Test accuracy, mean ± SD over seeds (red = FedAvg)")
        savefig(fig, f"{name}_baselines")

        all_md += md

        # ---------------------------------------------------------------- Results section (numbers only from data)
        acc, auc, f1m = (summarize(main_df[m].values) for m in ["accuracy", "roc_auc", "macro_f1"])
        target = "met" if acc["mean"] >= 0.94 else "NOT met"
        sec = [f"### Results — {name}", "",
               f"On the {name} data ({info['final_rows_before_encoding']} records after removing {info['duplicates_removed']} exact duplicates; "
               f"{info['n_train']}/{info['n_val']}/{info['n_test']} train/validation/test; majority-class test accuracy {info['majority_baseline_test_acc']:.3f}), "
               f"FedAvg with K={cfg['n_clients']} IID clients reached a test accuracy of {fmt(acc['mean'], acc['std'])} "
               f"(95% CI [{acc['ci_lo']:.4f}, {acc['ci_hi']:.4f}], n={acc['n']} seeds), ROC-AUC {fmt(auc['mean'], auc['std'])} and macro-F1 {fmt(f1m['mean'], f1m['std'])} "
               f"at the default 0.5 threshold. The 94% accuracy target was {target} on this dataset. Standard deviations are across independent seeds (split, partition, initialisation), not across test samples."]
        cen = base[base.setting == "Centralized MLP (non-FL)"]
        loc = base[base.setting == "Local-only (mean over clients)"]
        rf = base[base.setting == "Centralized Random Forest (non-FL)"]
        if len(cen):
            sec.append(f"The centralized (non-federated) MLP trained on the same records achieved {fmt(cen.accuracy.mean(), cen.accuracy.std(ddof=1))} accuracy and AUC {fmt(cen.roc_auc.mean(), cen.roc_auc.std(ddof=1))}; "
                       f"the centralized Random Forest {fmt(rf.accuracy.mean(), rf.accuracy.std(ddof=1))}; and local-only training (mean over the 10 clients) {fmt(loc.accuracy.mean(), loc.accuracy.std(ddof=1))}. "
                       f"Paired differences are in Table 3 (n={len(main_df)} seeds, so statistical power is limited).")
        if len(ni):
            parts = []
            for l in labs:
                d = ni[ni.setting == l]
                parts.append(f"{l}: accuracy {fmt(d.accuracy.mean(), d.accuracy.std(ddof=1))}, AUC {fmt(d.roc_auc.mean(), d.roc_auc.std(ddof=1))}")
            sec.append("Under label-skewed partitions, " + "; ".join(parts) + ". See Table 4 for training time, communication and client-level spread; robustness should only be claimed where these differences are small relative to the seed SD.")
        if len(sc):
            parts = []
            for K in ks:
                d = sc[sc.n_clients == K]
                parts.append(f"K={int(K)}: acc {fmt(d.accuracy.mean(), d.accuracy.std(ddof=1), 3)}, {d.total_comm_mb.mean():.2f} MB, {d.train_time.mean():.1f} s, {d.rounds_run.mean():.0f} rounds")
            sec.append("Scalability with the pooled training data held fixed: " + "; ".join(parts) + ".")
        all_md_res = sec
        results_json[name] = dict(config=cfg, fedavg=agg(main_df, MAIN_METRICS), target_94_met=bool(acc["mean"] >= 0.94))
        with open(os.path.join(C.RESULTS_DIR, f"results_section_{name}.md"), "w") as f:
            f.write("\n\n".join(sec))

    pd.DataFrame(summary_rows).to_csv(os.path.join(C.RESULTS_DIR, "metrics_summary.csv"), index=False)
    pd.DataFrame(seed_rows).to_csv(os.path.join(C.RESULTS_DIR, "seed_results.csv"), index=False)
    pd.DataFrame(scal_rows).to_csv(os.path.join(C.RESULTS_DIR, "scalability_results.csv"), index=False)
    pd.DataFrame(cm_rows).to_csv(os.path.join(C.RESULTS_DIR, "confusion_matrix.csv"), index=False)
    save_json(results_json, os.path.join(C.RESULTS_DIR, "results.json"))
    all_md.append("\n# Consistency checks\n" + "\n".join(f"- {n}: {k} -> {v}" for n, k, v in checks))
    open(os.path.join(C.RESULTS_DIR, "all_tables.md"), "w").write("\n".join(all_md))
    print("\n".join(f"{n}: {k} -> {v}" for n, k, v in checks))


if __name__ == "__main__":
    main()
