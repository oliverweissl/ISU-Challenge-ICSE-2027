"""Pure functions merging per-modality hints. hints: list (one per modality) of (B, L, D) tensors."""
import torch


def merge_sum(hints, w):
    return sum(wi * h for wi, h in zip(w, hints))


def merge_weighted(hints, w):
    """w: per-modality weights, either python floats or (B, L, 1) tensors."""
    return sum(wi * h for wi, h in zip(w, hints))


def project_conflicts(hints, thresh=0.0, strength=1.0, center=True):
    """PCGrad on tokens (two modalities): where cos(h0, h1) < thresh remove from each the component along the other."""
    h0, h1 = hints
    a, b = h0.float(), h1.float()
    # raw hints share a large common-mode component (mean cos ~0.8); measure/resolve conflict on the residual
    ca = a.mean(1, keepdim=True) if center else 0.0
    cb = b.mean(1, keepdim=True) if center else 0.0
    a, b = a - ca, b - cb
    dot = (a * b).sum(-1, keepdim=True)
    na2 = (a * a).sum(-1, keepdim=True).clamp_min(1e-8)
    nb2 = (b * b).sum(-1, keepdim=True).clamp_min(1e-8)
    cos = dot / (na2.sqrt() * nb2.sqrt())
    conflict = (cos < thresh).float() * strength
    a2 = a - conflict * (dot / nb2) * b
    b2 = b - conflict * (dot / na2) * a
    return [(a2 + ca).to(h0.dtype), (b2 + cb).to(h1.dtype)], (cos < thresh).float().mean().item(), cos.mean().item()
