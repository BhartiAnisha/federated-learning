"""Dataset loading, leakage-safe splitting/preprocessing, and client partitioning."""
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DATA_DIR, DATASETS, SPLIT


def load_dataset(name, drop_duration=False):
    """Return (X DataFrame, y ndarray {0,1}, meta dict). Positive class = the clinically / commercially
    'interesting' class (malignant, subscribed, indoor reading)."""
    path = os.path.join(DATA_DIR, DATASETS[name]["file"])
    meta = dict(name=name, source_file=DATASETS[name]["file"])
    if name == "hospital":
        raw = pd.read_csv(path)
        meta["original_shape"] = tuple(raw.shape)
        raw["Bare Nuclei"] = pd.to_numeric(raw["Bare Nuclei"], errors="coerce")   # '?' -> NaN
        meta["missing_values"] = int(raw.isna().sum().sum())
        y = (raw["Class"] == 4).astype(int)                                       # 4 = malignant
        X = raw.drop(columns=["Class", "Sample code number"])                     # ID is not a feature
        pos_name, neg_name = "malignant", "benign"
    elif name == "bank":
        raw = pd.read_csv(path, sep=";")
        meta["original_shape"] = tuple(raw.shape)
        meta["missing_values"] = int(raw.isna().sum().sum())
        y = (raw["y"] == "yes").astype(int)
        X = raw.drop(columns=["y"])
        if drop_duration:
            X = X.drop(columns=["duration"])
        pos_name, neg_name = "subscribed", "not subscribed"
    elif name == "iot":
        raw = pd.read_csv(path)
        meta["original_shape"] = tuple(raw.shape)
        meta["missing_values"] = int(raw.isna().sum().sum())
        raw = raw.drop_duplicates(subset="id")
        t = pd.to_datetime(raw["noted_date"], format="%d-%m-%Y %H:%M")
        X = pd.DataFrame({"temp": raw["temp"].astype(float), "hour": t.dt.hour, "minute": t.dt.minute,
                          "dayofweek": t.dt.dayofweek, "day": t.dt.day, "month": t.dt.month})
        y = (raw["out/in"] == "In").astype(int)            # In = positive (minority) class
        pos_name, neg_name = "In", "Out"
    else:
        raise ValueError(name)

    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    # remove exact duplicate records (features + label) so no copy can sit in train and test
    dup = pd.concat([X, y.rename("__y__")], axis=1).duplicated().values
    meta["duplicates_removed"] = int(dup.sum())
    X, y = X[~dup].reset_index(drop=True), y[~dup].reset_index(drop=True)
    meta["final_rows_before_encoding"] = int(len(X))
    meta["n_raw_features"] = int(X.shape[1])
    meta["classes"] = {neg_name: int((y == 0).sum()), pos_name: int((y == 1).sum())}
    meta["pos_name"], meta["neg_name"] = pos_name, neg_name
    return X, y.values.astype(int), meta


def make_preprocessor(X):
    cat = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) in ("string", "str", "category")]
    num = [c for c in X.columns if c not in cat]
    tfs = []
    if num:
        tfs.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num))
    if cat:
        tfs.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    return ColumnTransformer(tfs)


def split_and_preprocess(X, y, seed):
    """Stratified train/val/test split, preprocessing fitted on TRAIN ONLY."""
    Xtmp, Xte, ytmp, yte = train_test_split(X, y, test_size=SPLIT["test"], stratify=y, random_state=seed)
    val_frac = SPLIT["val"] / (1 - SPLIT["test"])
    Xtr, Xva, ytr, yva = train_test_split(Xtmp, ytmp, test_size=val_frac, stratify=ytmp, random_state=seed)
    pre = make_preprocessor(Xtr)
    A = pre.fit_transform(Xtr)
    return dict(Xtr=A.astype(np.float64), ytr=np.asarray(ytr), Xva=pre.transform(Xva).astype(np.float64),
                yva=np.asarray(yva), Xte=pre.transform(Xte).astype(np.float64), yte=np.asarray(yte),
                n_features=A.shape[1])


def partition_iid(y, n_clients, seed):
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(n_clients)]
    for c in np.unique(y):                       # stratified IID: every class spread evenly
        idx = rng.permutation(np.where(y == c)[0])
        for k, chunk in enumerate(np.array_split(idx, n_clients)):
            parts[k].extend(chunk.tolist())
    return [np.array(sorted(p), dtype=int) for p in parts]


def partition_dirichlet(y, n_clients, alpha, seed, min_size=5, max_tries=300):
    """Label-skew Dirichlet partition (Hsu et al., 2019): for every class, client shares ~ Dir(alpha)."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    for ms in (min_size, 2, 1):
        for _ in range(max_tries):
            parts = [[] for _ in range(n_clients)]
            for c in classes:
                idx = rng.permutation(np.where(y == c)[0])
                p = rng.dirichlet(alpha * np.ones(n_clients))
                cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
                for k, chunk in enumerate(np.split(idx, cuts)):
                    parts[k].extend(chunk.tolist())
            if min(len(p) for p in parts) >= ms:
                return [np.array(sorted(p), dtype=int) for p in parts]
    raise RuntimeError("could not build a Dirichlet partition")


def make_partition(y, n_clients, alpha, seed):
    return partition_iid(y, n_clients, seed) if alpha is None else partition_dirichlet(y, n_clients, alpha, seed)


def describe_partition(parts, y):
    sizes = [len(p) for p in parts]
    pos = [float(y[p].mean()) if len(p) else float("nan") for p in parts]
    return dict(n_clients=len(parts), size_min=min(sizes), size_max=max(sizes), size_mean=float(np.mean(sizes)),
                pos_rate_min=float(np.nanmin(pos)), pos_rate_max=float(np.nanmax(pos)), sizes=sizes, pos_rates=pos)
