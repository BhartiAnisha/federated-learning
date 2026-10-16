"""NumPy MLP (binary classifier) with manual back-propagation, plus Adam / SGD-momentum optimisers.
Logistic regression is the special case hidden=() ."""
import numpy as np


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class MLP:
    def __init__(self, in_dim, hidden=(32,), dropout=0.0, seed=0):
        rng = np.random.default_rng(seed)
        dims = [in_dim, *hidden, 1]
        self.dropout = float(dropout)
        self.W = [rng.standard_normal((dims[i], dims[i + 1])) * np.sqrt(2.0 / dims[i]) for i in range(len(dims) - 1)]
        self.b = [np.zeros(dims[i + 1]) for i in range(len(dims) - 1)]
        self.hidden = tuple(hidden)

    # ---- parameter handling -------------------------------------------------
    def params(self):
        out = []
        for W, b in zip(self.W, self.b):
            out += [W, b]
        return out

    @property
    def n_params(self):
        return int(sum(p.size for p in self.params()))

    def get_flat(self):
        return np.concatenate([p.ravel() for p in self.params()])

    def set_flat(self, v):
        i = 0
        for p in self.params():
            p[...] = v[i:i + p.size].reshape(p.shape)
            i += p.size

    # ---- forward / backward -------------------------------------------------
    def logits(self, X, train=False, rng=None, cache=None):
        a = X
        for i in range(len(self.W) - 1):
            z = a @ self.W[i] + self.b[i]
            h = np.maximum(z, 0.0)
            mask = None
            if train and self.dropout > 0:
                mask = (rng.random(h.shape) >= self.dropout) / (1.0 - self.dropout)
                h = h * mask
            if cache is not None:
                cache.append((a, z, mask))
            a = h
        if cache is not None:
            cache.append((a, None, None))
        return (a @ self.W[-1] + self.b[-1]).ravel()

    def predict_proba(self, X):
        return _sigmoid(self.logits(X))

    def loss_grads(self, X, y, sw, rng, wd=0.0, center=None, mu=0.0):
        """Weighted BCE (+ L2 weight decay + FedProx proximal term). Returns loss, grads (same order as params())."""
        cache = []
        z = self.logits(X, train=True, rng=rng, cache=cache)
        n = len(y)
        loss = np.sum(sw * (np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z))))) / n
        dz = (sw * (_sigmoid(z) - y) / n)[:, None]
        grads = [None] * (2 * len(self.W))
        for i in range(len(self.W) - 1, -1, -1):
            a_in, zpre, mask = cache[i]
            grads[2 * i] = a_in.T @ dz
            grads[2 * i + 1] = dz.sum(0)
            if i > 0:
                da = dz @ self.W[i].T
                pa, pz, pmask = cache[i - 1][0], cache[i - 1][1], cache[i - 1][2]
                # cache[i][0] is the (masked) activation of layer i-1; gradient flows through mask and ReLU
                if pmask is not None:
                    da = da * pmask
                dz = da * (pz > 0)
        if wd > 0:
            for i, W in enumerate(self.W):
                grads[2 * i] = grads[2 * i] + wd * W
                loss += 0.5 * wd * float(np.sum(W * W))
        if mu > 0 and center is not None:
            for g, p, c in zip(grads, self.params(), center):
                g += mu * (p - c)
        return float(loss), grads


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params, grads):
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


class SGD:
    def __init__(self, params, lr, momentum=0.9):
        self.lr, self.mom = lr, momentum
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params, grads):
        for p, g, v in zip(params, grads, self.v):
            v *= self.mom; v += g
            p -= self.lr * v


def make_optimizer(name, params, lr):
    return Adam(params, lr) if name == "adam" else SGD(params, lr)


def bce(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
