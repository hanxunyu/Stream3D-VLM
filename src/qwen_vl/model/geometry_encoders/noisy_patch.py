"""Noisy / ablated geometry-prior wrapper.

Replaces (or perturbs) `model.geometry_encoder.encode` so that the rest of the
pipeline (geometry_merger, concat_geometry_merger, feature_fusion) consumes a
noise tensor instead of the real StreamVGGT features. Used for the
"how much does the geometry prior matter?" ablation requested by reviewers.

CLI mode strings (exposed via `--noisy_geometry <spec>`):

    none                  -- (default) no patching, behave as usual
    zero                  -- features replaced by all-zeros (no info)
    random                -- features ~ N(0, 1)
    random_<std>          -- features ~ N(0, std), e.g. 'random_0.5'
    matched               -- features ~ N(mu, sigma) where (mu, sigma) is
                             calibrated from real StreamVGGT outputs on the
                             FIRST batch (most useful for paper ablations)
    perturb_<std>         -- real features + N(0, std), e.g. 'perturb_0.3'
    shuffle               -- real features but shuffled across the frame axis
                             (preserves marginal stats, breaks per-frame info)

The patcher returns the same `(features, concat_features)` shape and dtype as
the original encoder so downstream `feature_fusion` works without changes.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


# --------------------------------------------------------------------------- #
# CLI parsing                                                                  #
# --------------------------------------------------------------------------- #

def parse_noise_spec(spec: str) -> Dict:
    """Parse a noise-mode string into a config dict.

    Returns {'mode': 'none'} when spec is None / '' / 'none' / 'off'.
    """
    if spec is None:
        return {"mode": "none"}
    s = str(spec).strip().lower()
    if s in ("", "none", "off", "false", "no"):
        return {"mode": "none"}
    if s in ("zero", "zeros"):
        return {"mode": "zero"}
    if s == "matched":
        return {"mode": "matched"}
    if s == "shuffle":
        return {"mode": "shuffle"}
    if s == "random":
        return {"mode": "random", "std": 1.0}
    if s.startswith("random_"):
        return {"mode": "random", "std": float(s[len("random_"):])}
    if s == "perturb":
        return {"mode": "perturb", "std": 1.0}
    if s.startswith("perturb_"):
        return {"mode": "perturb", "std": float(s[len("perturb_"):])}
    raise ValueError(f"Unknown --noisy_geometry spec: {spec!r}")


# --------------------------------------------------------------------------- #
# Patching                                                                     #
# --------------------------------------------------------------------------- #

def patch_geometry_encoder_with_noise(
    model,
    spec: str = "matched",
    seed: int = 42,
    feature_dim: int = 2048,
    verbose: bool = True,
) -> bool:
    """Monkey-patch `model.geometry_encoder.encode` according to `spec`.

    Returns True if patching took effect, False if mode was 'none' or the
    model has no geometry encoder. Idempotent: a second call on the same
    encoder is a no-op.
    """
    cfg = parse_noise_spec(spec)
    mode = cfg["mode"]
    if mode == "none":
        if verbose:
            print(f"[NoisyPrior] Disabled (spec={spec!r}); using real geometry encoder.")
        return False

    if not hasattr(model, "geometry_encoder") or model.geometry_encoder is None:
        if verbose:
            print(f"[NoisyPrior] Model has no geometry_encoder; --noisy_geometry "
                  f"is a no-op.")
        return False

    encoder = model.geometry_encoder
    if getattr(encoder, "_noisy_patched", False):
        if verbose:
            print(f"[NoisyPrior] Already patched; skipping (existing mode kept).")
        return False

    patch_size = getattr(encoder, "patch_size", 14)

    original_encode = encoder.encode
    encoder._orig_encode_before_noise = original_encode

    state: Dict = {
        "calibrated": False,
        "feat_mean": None, "feat_std": None,
        "concat_mean": None, "concat_std": None,
        "calls": 0,
    }

    # Per-encoder noise generator → reproducible across runs, independent
    # of any user-side torch.manual_seed.
    if torch.cuda.is_available():
        gen_device = "cuda"
        gen = torch.Generator(device="cuda").manual_seed(seed)
    else:
        gen_device = "cpu"
        gen = torch.Generator(device="cpu").manual_seed(seed)

    def _randn(shape, mean: float, std: float, device, dtype):
        # torch.randn supports a 'generator' kwarg only when the generator's
        # device matches; if a sample lands on a different cuda device we fall
        # back to the global generator (still seeded above for determinism).
        try:
            t = torch.randn(*shape, generator=gen, device=device, dtype=torch.float32)
        except RuntimeError:
            t = torch.randn(*shape, device=device, dtype=torch.float32)
        return (t * std + mean).to(dtype)

    def _calibrate(images):
        feats, concat = original_encode(images)
        with torch.no_grad():
            state["feat_mean"] = float(feats.float().mean().item())
            state["feat_std"] = max(float(feats.float().std().item()), 1e-6)
            state["concat_mean"] = float(concat.float().mean().item())
            state["concat_std"] = max(float(concat.float().std().item()), 1e-6)
        state["calibrated"] = True
        if verbose:
            print(
                f"[NoisyPrior] Calibrated from first real batch: "
                f"feat   mu={state['feat_mean']:+.3f} sigma={state['feat_std']:.3f}; "
                f"concat mu={state['concat_mean']:+.3f} sigma={state['concat_std']:.3f}"
            )
        return feats, concat

    def noisy_encode(images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # images: (n_image, 3, H, W)
        n_image, _, H, W = images.shape
        Hp, Wp = H // patch_size, W // patch_size
        n_patches = Hp * Wp
        device = images.device
        # Match the dtype the original encoder produced (bf16 on Ampere+, fp16 otherwise).
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

        state["calls"] += 1

        if mode == "perturb":
            std = float(cfg.get("std", 1.0))
            feats, concat = original_encode(images)
            feats = feats + torch.randn_like(feats) * std
            concat = concat + torch.randn_like(concat) * std
            return feats, concat

        if mode == "shuffle":
            feats, concat = original_encode(images)
            if feats.shape[0] > 1:
                perm = torch.randperm(feats.shape[0], device=feats.device)
                feats = feats[perm]
                concat = concat[perm]
            return feats, concat

        if mode == "zero":
            feats = torch.zeros(n_image, n_patches, feature_dim, device=device, dtype=dtype)
            concat = torch.zeros(n_image, 1 + n_patches, feature_dim, device=device, dtype=dtype)
            return feats, concat

        if mode == "matched":
            if not state["calibrated"]:
                # Run real encoder once to learn (mu, sigma); return real outputs
                # for that single batch so the very first frame is not penalised.
                # All subsequent batches will be replaced with calibrated noise.
                return _calibrate(images)
            feats = _randn(
                (n_image, n_patches, feature_dim),
                state["feat_mean"], state["feat_std"], device, dtype,
            )
            concat = _randn(
                (n_image, 1 + n_patches, feature_dim),
                state["concat_mean"], state["concat_std"], device, dtype,
            )
            return feats, concat

        if mode == "random":
            std = float(cfg.get("std", 1.0))
            feats = _randn((n_image, n_patches, feature_dim), 0.0, std, device, dtype)
            concat = _randn((n_image, 1 + n_patches, feature_dim), 0.0, std, device, dtype)
            return feats, concat

        raise ValueError(f"Unknown noise mode: {mode}")

    encoder.encode = noisy_encode
    encoder._noisy_patched = True
    encoder._noisy_spec = spec
    encoder._noisy_cfg = cfg

    if verbose:
        print(f"[NoisyPrior] Patched geometry_encoder.encode with spec={spec!r} "
              f"(seed={seed}, feature_dim={feature_dim}, patch_size={patch_size})")

    return True
