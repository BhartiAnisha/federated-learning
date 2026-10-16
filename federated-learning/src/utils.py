import json, os, platform, random, sys, hashlib
import numpy as np
from scipy import stats


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def env_info():
    import sklearn, pandas, scipy, matplotlib
    return dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pandas.__version__,
                sklearn=sklearn.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__,
                os=platform.platform(), cpu_count=os.cpu_count(), machine=platform.machine())


def to_bytes(flat):
    """Actual serialised payload of a model update: float32 little-endian, no headers."""
    return np.asarray(flat, dtype=np.float32).tobytes()


def from_bytes(buf):
    return np.frombuffer(buf, dtype=np.float32).astype(np.float64)


def summarize(x):
    """Mean, sample std (ddof=1), min, max, n, 95% t-CI."""
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    n = len(x)
    if n == 0:
        return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan, n=0, ci_lo=np.nan, ci_hi=np.nan)
    mean = x.mean()
    std = x.std(ddof=1) if n > 1 else np.nan
    if n > 1:
        h = stats.t.ppf(0.975, n - 1) * std / np.sqrt(n)
    else:
        h = np.nan
    return dict(mean=mean, std=std, min=x.min(), max=x.max(), n=n, ci_lo=mean - h, ci_hi=mean + h)


def paired_test(a, b):
    """Paired t-test and Cohen's d_z for two equal-length arrays of per-seed metrics."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if len(d) < 2 or np.allclose(d.std(ddof=1), 0):
        return dict(t=np.nan, p=np.nan, dz=np.nan, mean_diff=d.mean() if len(d) else np.nan, n=len(d))
    t, p = stats.ttest_rel(a, b)
    return dict(t=float(t), p=float(p), dz=float(d.mean() / d.std(ddof=1)), mean_diff=float(d.mean()), n=len(d))


def fmt(mean, std, nd=4):
    if mean is None or not np.isfinite(mean):
        return "n/a"
    if std is None or not np.isfinite(std):
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=NpEncoder)
