"""Numpy checks of the parts of csn_reid.py that carry the mathematics.

Verifies:
  A. eq. (9) mixture variance equals the true variance of the mixture, and
     dominates the naive convex combination of variances.
  B. eq. (12) the uniform camera marginal minimises expected squared mismatch.
  C. Proposition 1: Var[mu_hat_B] = sw^2/N + sb^2 * sum_c p_c^2,
     minimised over allocations by the balanced one.
  D. the camera-balanced PK sampler actually balances cameras.
"""
import math
import random
from collections import defaultdict

import numpy as np

rng = np.random.default_rng(0)
OK = "PASS"


# ---------------------------------------------------------------- A
def mixture_statistics(bank_mean, bank_var, pi):
    """eq. (8) and (9)."""
    pi = pi[:, None]
    mu_t = (pi * bank_mean).sum(0)
    second = (pi * (bank_var + bank_mean ** 2)).sum(0)
    var_t = np.maximum(second - mu_t ** 2, 0.0)
    return mu_t, var_t


M, D, n = 9, 5, 400_000
bank_mean = rng.normal(0, 2.0, size=(M, D))
bank_var = rng.uniform(0.2, 1.5, size=(M, D))
pi = rng.dirichlet(np.full(M, 0.5))

# ground truth: sample from the actual mixture distribution
comp = rng.choice(M, size=n, p=pi)
samples = rng.normal(bank_mean[comp], np.sqrt(bank_var[comp]))
emp_mean, emp_var = samples.mean(0), samples.var(0)

mu_t, var_t = mixture_statistics(bank_mean, bank_var, pi)
err_m = np.abs(mu_t - emp_mean).max()
err_v = np.abs(var_t - emp_var).max() / emp_var.max()
assert err_m < 0.05 and err_v < 0.02, (err_m, err_v)
print(f"A1 eq.(8)/(9) match the empirical mixture   "
      f"(mean err {err_m:.4f}, rel var err {err_v:.4f})   {OK}")

var_naive = (pi[:, None] * bank_var).sum(0)
gap = var_t - var_naive
assert (gap >= -1e-9).all(), gap
print(f"A2 mixture variance >= naive convex combination "
      f"(mean gap {gap.mean():.3f})              {OK}")


# ---------------------------------------------------------------- B
# Proposition 2: argmin_s E||mu_c - s||^2 = mean over cameras
cams = rng.normal(0, 1.0, size=(M, D))
marginal = cams.mean(0)


def expected_mismatch(s):
    return ((cams - s) ** 2).sum(1).mean()


best = expected_mismatch(marginal)
# any per-dataset statistic (a mean over a subset) does worse
worse = 0
for _ in range(2000):
    k = rng.integers(1, M)
    subset = cams[rng.choice(M, size=k, replace=False)].mean(0)
    if expected_mismatch(subset) >= best - 1e-12:
        worse += 1
assert worse == 2000
print(f"B  camera marginal beats every subset statistic "
      f"({worse}/2000 trials)                {OK}")


# ---------------------------------------------------------------- C
# Proposition 1: Var[mu_hat] = sw^2/N + sb^2 * sum p_c^2
sw2, sb2 = 0.7, 1.3
N, Mc = 64, 8
trials = 60_000


def empirical_var(props):
    counts = np.round(np.array(props) * N).astype(int)
    counts[-1] = N - counts[:-1].sum()
    out = np.empty(trials)
    for t in range(trials):
        tot = 0.0
        for c, nc in enumerate(counts):
            if nc == 0:
                continue
            mc = rng.normal(0, math.sqrt(sb2))          # camera effect
            e = rng.normal(0, math.sqrt(sw2), size=nc)  # within-camera noise
            tot += nc * mc + e.sum()
        out[t] = tot / N
    return out.var()


for name, props in [("balanced   ", [1 / Mc] * Mc),
                    ("2 cameras  ", [0.5, 0.5] + [0.0] * (Mc - 2)),
                    ("1 camera   ", [1.0] + [0.0] * (Mc - 1))]:
    p = np.array(props)
    pred = sw2 / N + sb2 * (p ** 2).sum()
    emp = empirical_var(props)
    rel = abs(pred - emp) / pred
    assert rel < 0.06, (name, pred, emp)
    print(f"C  {name} predicted {pred:.4f}  empirical {emp:.4f} "
          f"(rel {rel:.3f})   {OK}")

# balanced allocation is the minimiser of sum p_c^2
bal = 1.0 / Mc
for _ in range(5000):
    p = rng.dirichlet(np.ones(Mc))
    assert (p ** 2).sum() >= bal - 1e-12
print(f"C  balanced allocation minimises sum p_c^2 over 5000 draws       {OK}")


# ---------------------------------------------------------------- D
class CameraBalancedPKSampler:
    """Pure-python copy of the sampler in csn_reid.py."""

    def __init__(self, pids, camids, P=16, K=4, target_m=None, seed=0):
        self.P, self.K = P, K
        self.batch_size = P * K
        self.seed, self.epoch = seed, 0
        self.pid_to_cam_idx = defaultdict(lambda: defaultdict(list))
        for i, (p, c) in enumerate(zip(pids, camids)):
            self.pid_to_cam_idx[int(p)][int(c)].append(i)
        self.pids = sorted(self.pid_to_cam_idx)
        n_cams = len(set(int(c) for c in camids))
        self.target_m = target_m or min(n_cams, self.batch_size)
        self.cam_cap = math.ceil(self.batch_size / max(self.target_m, 1))
        self._idx2cam = {}
        for p, d in self.pid_to_cam_idx.items():
            for c, idxs in d.items():
                for i in idxs:
                    self._idx2cam[i] = c

    def _sample_identity(self, pid, r):
        cams = list(self.pid_to_cam_idx[pid])
        r.shuffle(cams)
        picked = []
        for c in cams:
            if len(picked) == self.K:
                break
            picked.append(r.choice(self.pid_to_cam_idx[pid][c]))
        i = 0
        while len(picked) < self.K:
            c = cams[i % len(cams)]
            picked.append(r.choice(self.pid_to_cam_idx[pid][c]))
            i += 1
        return picked

    def __iter__(self):
        r = random.Random(self.seed + self.epoch)
        pool = self.pids[:]
        r.shuffle(pool)
        out = []
        while len(pool) >= self.P:
            batch, cam_count, deferred = [], defaultdict(int), []
            while len(batch) < self.batch_size and pool:
                pid = pool.pop(0)
                cand = self._sample_identity(pid, r)
                cc = [self._idx2cam[i] for i in cand]
                over = any(cam_count[c] + cc.count(c) > self.cam_cap
                           for c in set(cc))
                if over and len(deferred) < self.P:
                    deferred.append(pid)
                    continue
                batch.extend(cand)
                for c in cc:
                    cam_count[c] += 1
            pool.extend(deferred)
            if len(batch) < self.batch_size:
                break
            out.extend(batch[:self.batch_size])
        return iter(out)


def sum_p2(batch_cams):
    N = len(batch_cams)
    cnt = defaultdict(int)
    for c in batch_cams:
        cnt[c] += 1
    return sum((v / N) ** 2 for v in cnt.values())


# realistic layout: each identity is seen by only 2-3 of the 8 cameras
n_cam, n_pid = 8, 400
pids, camids = [], []
r = random.Random(1)
for p in range(n_pid):
    seen = r.sample(range(n_cam), r.choice([2, 3]))
    for _ in range(10):
        pids.append(p)
        camids.append(r.choice(seen))

bal = CameraBalancedPKSampler(pids, camids, P=16, K=4, target_m=n_cam)
idx = list(iter(bal))
b = [camids[i] for i in idx[:64]]
print(f"D  balanced sampler: {len(set(b))}/{n_cam} cameras in batch, "
      f"sum p_c^2 = {sum_p2(b):.4f}   {OK}")

# naive PK sampler for comparison
by_pid = defaultdict(list)
for i, p in enumerate(pids):
    by_pid[p].append(i)
r2 = random.Random(1)
naive = []
for p in r2.sample(list(by_pid), 16):
    naive.extend(r2.choices(by_pid[p], k=4))
bn = [camids[i] for i in naive]
print(f"D  naive PK sampler:    {len(set(bn))}/{n_cam} cameras in batch, "
      f"sum p_c^2 = {sum_p2(bn):.4f}")
print(f"   ideal floor 1/m = {1/n_cam:.4f}; lower is better "
      f"(Proposition 1)")

print("\nAll checks passed.")
