"""
Camera-Statistic Normalization (CSN) for domain-generalizable person ReID.

Reference implementation of the method described in

    Idrissi Alami M., Ez-zahout A., Omary F.
    "Camera-Statistic Normalization for Domain-Generalizable Person
     Re-identification in Unseen Surveillance Networks."

Contents
    CSN2d                  - the normalization layer          (Sec. 4.1-4.3, eqs. 6-10)
    convert_to_csn         - swaps BatchNorm2d for CSN2d in any backbone
    csn_virtual_camera     - Dirichlet draw of a virtual camera (eq. 7)
    csn_set_inference_mode - marginal-statistic inference       (Sec. 4.6, eq. 12)
    CameraBalancedPKSampler- batch construction                 (Sec. 4.5)
    statistic_perturbation_consistency - the L_spc term         (eq. 11)
    train_one_epoch        - the loop of Listing 1

Requires: torch >= 1.12, torchvision.

The code is deliberately self-contained and framework-agnostic: it does not
depend on a ReID toolbox, so it can be dropped into an existing pipeline.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler


# ======================================================================
#  1. The CSN layer
# ======================================================================
class CSN2d(nn.Module):
    """Camera-Statistic Normalization.

    Replaces a BatchNorm2d. Standardises with a per-channel convex blend of
    (a) the instance statistics of each sample and (b) the statistics of a
    *virtual camera* obtained by Dirichlet mixing of a bank of per-camera
    population statistics.

        mu_csn  = (1 - lam) * mu_in  + lam * mu_tilde              (eq. 10)
        var_csn = (1 - lam) * var_in + lam * var_tilde

    The virtual camera (mu_tilde, var_tilde) is *not* computed here: it is
    injected from outside by `csn_virtual_camera` so that every CSN layer in
    one forward pass shares the same Dirichlet draw (Sec. 4.2, final para).

    Args
        num_features : number of channels D
        num_cameras  : number of source cameras M
        momentum     : 1 - rho of eq. (6); EMA rate for the bank
        affine       : if True the affine parameters are learnable. The paper
                       freezes them (Sec. 4.3), which is affine=True +
                       requires_grad_(False), handled by `freeze_affine`.
    """

    def __init__(
        self,
        num_features: int,
        num_cameras: int,
        eps: float = 1e-5,
        momentum: float = 0.1,          # = 1 - rho, rho = 0.9
        affine: bool = True,
        freeze_affine: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_cameras = num_cameras
        self.eps = eps
        self.momentum = momentum

        # --- affine parameters (eq. 3) ---------------------------------
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
            if freeze_affine:                       # Domain Frozen, Sec. 4.3
                self.weight.requires_grad_(False)
                self.bias.requires_grad_(False)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        # --- per-channel gate lambda, eq. (10) -------------------------
        # lam = sigmoid(rho_param); initialised at 0 -> lam = 0.5
        self.gate_logit = nn.Parameter(torch.zeros(num_features))

        # --- camera statistics bank, Sec. 4.1 --------------------------
        # buffers, not parameters: they receive no gradient
        self.register_buffer("bank_mean", torch.zeros(num_cameras, num_features))
        self.register_buffer("bank_var", torch.ones(num_cameras, num_features))
        self.register_buffer("bank_seen",
                             torch.zeros(num_cameras, dtype=torch.bool))

        # --- transient state, set by csn_virtual_camera ----------------
        self._virtual: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._update_bank: bool = True

    # ------------------------------------------------------------------
    def set_virtual(self, mean: Optional[torch.Tensor],
                    var: Optional[torch.Tensor]) -> None:
        self._virtual = None if mean is None else (mean, var)

    def set_update_bank(self, flag: bool) -> None:
        self._update_bank = flag

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_bank_from(self, x: torch.Tensor,
                          camera_ids: torch.Tensor) -> None:
        """EMA update of the bank, eq. (6). Only cameras present in the
        mini-batch are touched."""
        for c in camera_ids.unique():
            sel = camera_ids == c
            xs = x[sel]                                   # [n_c, D, H, W]
            mu = xs.mean(dim=(0, 2, 3))
            var = xs.var(dim=(0, 2, 3), unbiased=False)
            ci = int(c)
            if not bool(self.bank_seen[ci]):
                # first sight of this camera: initialise rather than blend,
                # otherwise the zero/one initial value biases the estimate
                self.bank_mean[ci] = mu
                self.bank_var[ci] = var
                self.bank_seen[ci] = True
            else:
                m = self.momentum
                self.bank_mean[ci] = (1 - m) * self.bank_mean[ci] + m * mu
                self.bank_var[ci] = (1 - m) * self.bank_var[ci] + m * var

    # ------------------------------------------------------------------
    def mixture_statistics(self, pi: torch.Tensor
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Virtual-camera statistics for Dirichlet weights `pi` [M].

        mu_t  = sum_c pi_c mu_c                                     (eq. 8)
        var_t = sum_c pi_c (var_c + mu_c^2) - mu_t^2                (eq. 9)

        Note the law-of-total-variance form: the naive convex combination of
        variances would ignore the dispersion of the component means and
        systematically understate the spread (Sec. 4.2).
        """
        pi = pi.to(self.bank_mean.device).unsqueeze(1)          # [M, 1]
        mu_t = (pi * self.bank_mean).sum(0)                     # [D]
        second = (pi * (self.bank_var + self.bank_mean ** 2)).sum(0)
        var_t = (second - mu_t ** 2).clamp_min(0.0)
        return mu_t, var_t

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor,
                camera_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # camera ids are threaded through a module attribute rather than the
        # signature, so that CSN2d stays a drop-in replacement for BatchNorm2d
        if camera_ids is None:
            camera_ids = getattr(self, "_camera_ids", None)

        if self.training and self._update_bank and camera_ids is not None:
            self._update_bank_from(x.detach(), camera_ids)

        # instance statistics, eq. (1)
        mu_in = x.mean(dim=(2, 3), keepdim=True)                # [N, D, 1, 1]
        var_in = x.var(dim=(2, 3), keepdim=True, unbiased=False)

        if self._virtual is None:
            # no virtual camera supplied -> pure instance normalization
            mu, var = mu_in, var_in
        else:
            mu_t, var_t = self._virtual                          # [D], [D]
            lam = torch.sigmoid(self.gate_logit).view(1, -1, 1, 1)
            mu_t = mu_t.view(1, -1, 1, 1)
            var_t = var_t.view(1, -1, 1, 1)
            mu = (1 - lam) * mu_in + lam * mu_t                  # eq. (10)
            var = (1 - lam) * var_in + lam * var_t

        out = (x - mu) / torch.sqrt(var + self.eps)
        if self.weight is not None:
            out = out * self.weight.view(1, -1, 1, 1) \
                + self.bias.view(1, -1, 1, 1)
        return out

    def extra_repr(self) -> str:
        return (f"{self.num_features}, num_cameras={self.num_cameras}, "
                f"eps={self.eps}, momentum={self.momentum}")


# ======================================================================
#  2. Backbone surgery and per-pass control
# ======================================================================
def convert_to_csn(module: nn.Module, num_cameras: int,
                   stages: Optional[Sequence[str]] = None,
                   momentum: float = 0.1,
                   freeze_affine: bool = True) -> nn.Module:
    """Recursively replace every BatchNorm2d by a CSN2d, copying the
    ImageNet affine parameters across.

    `stages` optionally restricts the replacement to submodules whose
    qualified name starts with one of the given prefixes, e.g.
    ("layer3", "layer4") for the Stage 3-4 ablation of Table 6.
    """
    def _convert(mod: nn.Module, prefix: str = "") -> nn.Module:
        for name, child in mod.named_children():
            qual = f"{prefix}{name}"
            if isinstance(child, nn.BatchNorm2d):
                if stages is not None and not any(
                        qual.startswith(s) for s in stages):
                    continue
                new = CSN2d(child.num_features, num_cameras,
                            eps=child.eps, momentum=momentum,
                            affine=child.affine,
                            freeze_affine=freeze_affine)
                if child.affine:
                    with torch.no_grad():
                        new.weight.copy_(child.weight)
                        new.bias.copy_(child.bias)
                setattr(mod, name, new)
            else:
                _convert(child, qual + ".")
        return mod

    return _convert(module)


def csn_layers(model: nn.Module) -> List[CSN2d]:
    return [m for m in model.modules() if isinstance(m, CSN2d)]


def csn_set_camera_ids(model: nn.Module, camera_ids: torch.Tensor) -> None:
    for m in csn_layers(model):
        m._camera_ids = camera_ids


def csn_virtual_camera(model: nn.Module, alpha: float = 0.5,
                       cameras_per_draw: int = 8,
                       generator: Optional[torch.Generator] = None) -> None:
    """Draw ONE Dirichlet weight vector (eq. 7) and install the corresponding
    virtual camera in every CSN layer.

    The same `pi` is shared across depth, so the perturbation is a coherent
    change of style rather than uncorrelated jitter (Sec. 4.2).
    Only cameras already observed are eligible.
    """
    layers = csn_layers(model)
    if not layers:
        return
    ref = layers[0]
    seen = torch.nonzero(ref.bank_seen, as_tuple=False).flatten()
    if seen.numel() == 0:                 # bank still empty (first steps)
        for m in layers:
            m.set_virtual(None, None)
        return

    k = min(cameras_per_draw, seen.numel())
    idx = seen[torch.randperm(seen.numel(), generator=generator)[:k]]

    conc = torch.full((k,), float(alpha))
    w = torch._sample_dirichlet(conc)                       # [k]

    pi = torch.zeros(ref.num_cameras)
    pi[idx.cpu()] = w

    for m in layers:
        mu_t, var_t = m.mixture_statistics(pi)
        m.set_virtual(mu_t, var_t)


def csn_set_inference_mode(model: nn.Module) -> None:
    """Marginal-statistic inference, Proposition 2 / eq. (12).

    Installs the uniform mixture over all observed cameras. After this call
    the layer is a deterministic affine normalization with precomputed
    constants: inference cost equals that of the unmodified backbone.
    """
    for m in csn_layers(model):
        seen = m.bank_seen.float()
        if seen.sum() == 0:
            m.set_virtual(None, None)
            continue
        pi = seen / seen.sum()
        mu_t, var_t = m.mixture_statistics(pi)
        m.set_virtual(mu_t, var_t)
        m.set_update_bank(False)


def csn_set_update_bank(model: nn.Module, flag: bool) -> None:
    for m in csn_layers(model):
        m.set_update_bank(flag)


# ======================================================================
#  3. Camera-balanced PK sampler  (Sec. 4.5, from Proposition 1)
# ======================================================================
class CameraBalancedPKSampler(Sampler):
    """PK sampler that maximises the number of distinct cameras per batch.

    Proposition 1:  Var[mu_hat] = sigma_w^2 / N + sigma_b^2 * sum_c p_c^2
    The second term falls only as the camera count m grows, and is minimised
    when the cameras are equally represented. This sampler therefore

      * within an identity, covers as many distinct cameras as that identity
        offers before re-using any camera;
      * across the batch, admits identities greedily under a cap on the share
        of any single camera (the balanced share, rounded up).

    Args
        pids       : identity label of every dataset index
        camids     : camera label of every dataset index
        P, K       : identities per batch, images per identity
        target_m   : cameras to aim for per batch; None -> as many as possible
    """

    def __init__(self, pids: Sequence[int], camids: Sequence[int],
                 P: int = 16, K: int = 4, target_m: Optional[int] = None,
                 seed: int = 0) -> None:
        assert len(pids) == len(camids)
        self.P, self.K = P, K
        self.batch_size = P * K
        self.seed = seed
        self.epoch = 0

        self.pid_to_cam_idx: Dict[int, Dict[int, List[int]]] = defaultdict(
            lambda: defaultdict(list))
        for i, (p, c) in enumerate(zip(pids, camids)):
            self.pid_to_cam_idx[int(p)][int(c)].append(i)
        self.pids = sorted(self.pid_to_cam_idx)

        n_cams = len(set(int(c) for c in camids))
        self.target_m = target_m if target_m is not None else min(n_cams,
                                                                  self.batch_size)
        # cap on images from any one camera, per Proposition 1
        self.cam_cap = math.ceil(self.batch_size / max(self.target_m, 1))
        self.length = (len(self.pids) // P) * self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    # ------------------------------------------------------------------
    def _sample_identity(self, pid: int, rng: random.Random) -> List[int]:
        """K images of `pid`, covering as many cameras as possible."""
        cams = list(self.pid_to_cam_idx[pid])
        rng.shuffle(cams)
        picked: List[int] = []
        # first pass: one image per camera
        for c in cams:
            if len(picked) == self.K:
                break
            picked.append(rng.choice(self.pid_to_cam_idx[pid][c]))
        # second pass: top up, re-using cameras round-robin
        i = 0
        while len(picked) < self.K:
            c = cams[i % len(cams)]
            picked.append(rng.choice(self.pid_to_cam_idx[pid][c]))
            i += 1
        return picked

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        pool = self.pids[:]
        rng.shuffle(pool)

        out: List[int] = []
        while len(pool) >= self.P:
            batch: List[int] = []
            cam_count: Dict[int, int] = defaultdict(int)
            deferred: List[int] = []

            while len(batch) < self.batch_size and pool:
                pid = pool.pop(0)
                cand = self._sample_identity(pid, rng)
                cand_cams = [self._camera_of(i) for i in cand]
                # greedy admission under the per-camera cap
                over = any(cam_count[c] + cand_cams.count(c) > self.cam_cap
                           for c in set(cand_cams))
                if over and len(deferred) < self.P:
                    deferred.append(pid)          # try again in a later batch
                    continue
                batch.extend(cand)
                for c in cand_cams:
                    cam_count[c] += 1

            pool.extend(deferred)
            if len(batch) < self.batch_size:
                break
            out.extend(batch[:self.batch_size])

        return iter(out)

    def _camera_of(self, idx: int) -> int:
        if not hasattr(self, "_idx2cam"):
            self._idx2cam = {}
            for p, d in self.pid_to_cam_idx.items():
                for c, idxs in d.items():
                    for i in idxs:
                        self._idx2cam[i] = c
        return self._idx2cam[idx]

    def __len__(self) -> int:
        return self.length


# ======================================================================
#  4. Losses
# ======================================================================
def statistic_perturbation_consistency(f1: torch.Tensor,
                                       f2: torch.Tensor) -> torch.Tensor:
    """L_spc, eq. (11): 1 - cosine between the two stochastic embeddings."""
    return (1.0 - F.cosine_similarity(f1, f2, dim=1)).mean()


def batch_hard_triplet(emb: torch.Tensor, labels: torch.Tensor,
                       margin: Optional[float] = None) -> torch.Tensor:
    """Batch-hard triplet loss. `margin=None` -> soft-margin variant."""
    d = torch.cdist(emb, emb, p=2)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)

    pos = d.masked_fill(~same | eye, float("-inf")).max(1).values
    neg = d.masked_fill(same, float("inf")).min(1).values

    if margin is None:
        return F.softplus(pos - neg).mean()
    return F.relu(pos - neg + margin).mean()


class LabelSmoothingCE(nn.Module):
    def __init__(self, eps: float = 0.1) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        n = logits.size(1)
        logp = F.log_softmax(logits, dim=1)
        nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
        smooth = -logp.mean(1)
        return ((1 - self.eps) * nll + self.eps * smooth).mean()


# ======================================================================
#  5. Training step  (Listing 1 of the paper)
# ======================================================================
def train_one_epoch(model: nn.Module, classifier: nn.Module,
                    loader, optimiser, device: torch.device,
                    alpha: float = 0.5, eta: float = 0.5,
                    cameras_per_draw: int = 8) -> Dict[str, float]:
    """One epoch of CSN training.

    `model`      : backbone returning an embedding [N, dim]
    `classifier` : linear layer embedding -> identity logits
    `loader`     : yields (images, pids, camids)
    """
    model.train()
    classifier.train()
    ce = LabelSmoothingCE()
    totals = defaultdict(float)
    nb = 0

    for images, pids, camids in loader:
        images = images.to(device, non_blocking=True)
        pids = pids.to(device, non_blocking=True)
        camids = camids.to(device, non_blocking=True)

        csn_set_camera_ids(model, camids)

        # ---- pass 1: update the bank, first virtual camera -------------
        csn_set_update_bank(model, True)
        csn_virtual_camera(model, alpha, cameras_per_draw)
        f1 = model(images)
        logits = classifier(f1)

        # ---- pass 2: independent virtual camera, bank frozen -----------
        csn_set_update_bank(model, False)
        csn_virtual_camera(model, alpha, cameras_per_draw)
        f2 = model(images)

        # ---- objective, eq. (13) ---------------------------------------
        l_id = ce(logits, pids)
        l_tri = batch_hard_triplet(f1, pids)
        l_spc = statistic_perturbation_consistency(f1, f2)
        loss = l_id + l_tri + eta * l_spc

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

        totals["loss"] += float(loss)
        totals["id"] += float(l_id)
        totals["tri"] += float(l_tri)
        totals["spc"] += float(l_spc)
        nb += 1

    return {k: v / max(nb, 1) for k, v in totals.items()}


@torch.no_grad()
def extract_features(model: nn.Module, loader,
                     device: torch.device) -> Tuple[torch.Tensor, ...]:
    """Deterministic inference with the marginal statistic (Proposition 2)."""
    model.eval()
    csn_set_inference_mode(model)
    feats, pids, camids = [], [], []
    for images, p, c in loader:
        f = model(images.to(device, non_blocking=True))
        feats.append(F.normalize(f, dim=1).cpu())
        pids.append(p)
        camids.append(c)
    return torch.cat(feats), torch.cat(pids), torch.cat(camids)


# ======================================================================
#  6. Self-test
# ======================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    import torchvision

    M = 9                       # source cameras, e.g. Market(6)+CS(1)+C3(2)
    net = torchvision.models.resnet50(weights=None)
    net.fc = nn.Identity()
    convert_to_csn(net, num_cameras=M)
    n_csn = len(csn_layers(net))
    print(f"CSN layers installed: {n_csn}")

    x = torch.randn(8, 3, 256, 128)
    cams = torch.randint(0, M, (8,))

    net.train()
    csn_set_camera_ids(net, cams)
    csn_set_update_bank(net, True)
    csn_virtual_camera(net)
    f1 = net(x)
    csn_set_update_bank(net, False)
    csn_virtual_camera(net)
    f2 = net(x)
    print("embeddings:", tuple(f1.shape),
          "| L_spc =", float(statistic_perturbation_consistency(f1, f2)))

    # mixture variance must dominate the naive convex combination (eq. 9)
    layer = csn_layers(net)[0]
    pi = torch.zeros(M); pi[0] = pi[1] = 0.5
    _, var_mix = layer.mixture_statistics(pi)
    var_naive = (pi.unsqueeze(1) * layer.bank_var).sum(0)
    print("mean(var_mixture - var_naive) =",
          float((var_mix - var_naive).mean()), ">= 0 as required")

    # determinism of marginal inference
    net.eval()
    csn_set_inference_mode(net)
    a, b = net(x), net(x)
    print("inference deterministic:", torch.allclose(a, b))

    # sampler
    pids = [i // 8 for i in range(800)]
    camids = [i % M for i in range(800)]
    s = CameraBalancedPKSampler(pids, camids, P=16, K=4, target_m=M)
    idx = list(iter(s))
    first = idx[:64]
    print("cameras in first batch:",
          len(set(camids[i] for i in first)), "of", M,
          "| per-camera cap:", s.cam_cap)
