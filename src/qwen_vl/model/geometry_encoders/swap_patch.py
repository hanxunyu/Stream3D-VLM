"""Replace ``model.geometry_encoder`` *at test time* with a different
reconstruction encoder (Pi3, VGGT, ...).

This is the structured counterpart to ``noisy_patch.py``: instead of
randomising the geometry features, we keep the trained ``geometry_merger``,
``concat_geometry_merger`` and ``feature_fusion`` modules untouched and only
change which encoder produces the features fed into them. The downstream
mergers were trained on StreamVGGT's feature distribution, so this is an
"how much does the prior encoder shape matter (zero-shot)?" ablation -- not
a fair head-to-head comparison (for that you'd need to retrain the mergers).

Typical use (after ``model = ...from_pretrained(...)``):

    from qwen_vl.model.geometry_encoders import swap_geometry_encoder
    swap_geometry_encoder(
        model,
        encoder_type="pi3",
        model_path="/path/to/Pi3",
    )
"""

from __future__ import annotations

from typing import Optional

import torch

from .factory import create_geometry_encoder


def _model_device(model) -> torch.device:
    """Best-effort device discovery for an arbitrary nn.Module."""
    for p in model.parameters():
        return p.device
    for b in model.buffers():
        return b.device
    return torch.device("cpu")


def _model_dtype(model) -> torch.dtype:
    """Best-effort dtype discovery (we use it to align the new encoder)."""
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float32


def swap_geometry_encoder(
    model,
    encoder_type: str,
    model_path: Optional[str] = None,
    reference_frame: str = "first",
    freeze: bool = True,
    train_or_eval_mode: str = "eval",
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    expected_feature_dim: Optional[int] = 2048,
    verbose: bool = True,
) -> bool:
    """Replace ``model.geometry_encoder`` with a fresh encoder built via the
    factory, optionally loading pretrained weights.

    Args:
        model: A ``Qwen2_5_VLForConditionalGenerationWithVGGT``-like model
            instance that exposes a ``geometry_encoder`` attribute.
        encoder_type: One of ``{"streamvggt", "vggt", "pi3", "cut3r"}``.
        model_path: Local dir or HF Hub id passed to the encoder's
            ``load_model``.
        reference_frame: ``"first"`` or ``"last"`` (only used by VGGT /
            StreamVGGT, ignored by Pi3).
        freeze: Freeze encoder parameters (recommended for inference).
        train_or_eval_mode: ``"train"`` / ``"eval"`` (used by StreamVGGT).
        device / dtype: Where to place the new encoder. Defaults to the
            existing encoder's (or model's) device / dtype.
        expected_feature_dim: If set, assert the new encoder's
            ``get_feature_dim()`` matches the trained
            ``geometry_merger.input_dim / spatial_merge_size**2`` channel
            count (default 2048 for the Stream3D-LLM mergers). Pass ``None``
            to skip the check.
        verbose: Print a one-line summary.

    Returns:
        ``True`` if the swap took effect, ``False`` if ``encoder_type`` was
        ``None`` / ``"none"`` / ``"off"``.
    """
    if encoder_type is None:
        return False
    et = str(encoder_type).strip().lower()
    if et in ("", "none", "off", "false", "no"):
        if verbose:
            print(f"[SwapEncoder] Disabled (encoder_type={encoder_type!r}); "
                  f"keeping the existing geometry_encoder.")
        return False

    if not hasattr(model, "geometry_encoder") or model.geometry_encoder is None:
        if verbose:
            print(f"[SwapEncoder] Model has no geometry_encoder; nothing to swap.")
        return False

    old_encoder = model.geometry_encoder
    if device is None:
        device = _model_device(old_encoder)
    if dtype is None:
        dtype = _model_dtype(old_encoder)
        # Encoders typically run their own internal autocast and keep their
        # *parameters* in fp32 for stability on Ampere+. Match the model's
        # dtype only if it's fp32 / bf16 / fp16; otherwise fall back to bf16.
        if dtype not in (torch.float32, torch.bfloat16, torch.float16):
            dtype = torch.bfloat16

    if verbose:
        old_name = type(old_encoder).__name__
        print(f"[SwapEncoder] Replacing {old_name} -> '{et}' "
              f"(model_path={model_path!r}, device={device}, dtype={dtype})")

    new_encoder = create_geometry_encoder(
        encoder_type=et,
        model_path=model_path,
        reference_frame=reference_frame,
        freeze_encoder=freeze,
        train_or_eval_mode=train_or_eval_mode,
    )

    if model_path is not None:
        new_encoder.load_model(model_path)
        if freeze:
            for p in new_encoder.parameters():
                p.requires_grad = False

    if expected_feature_dim is not None:
        got = new_encoder.get_feature_dim()
        if got != expected_feature_dim:
            raise ValueError(
                f"[SwapEncoder] {et!r} encoder feature_dim={got} but the "
                f"trained mergers expect {expected_feature_dim}. Either pass "
                f"a model whose feature_dim matches, or set "
                f"expected_feature_dim=None to bypass this check (the merger "
                f"linear will then fail at runtime)."
            )

    new_encoder = new_encoder.to(device=device, dtype=dtype).eval()

    # Drop the old encoder to free GPU memory before we attach the new one.
    del model.geometry_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.geometry_encoder = new_encoder

    # Most code paths read ``self.geometry_encoder.patch_size`` to compute
    # the patch grid -- keep it consistent.
    if not hasattr(new_encoder, "patch_size"):
        new_encoder.patch_size = 14

    # Tag for downstream introspection (e.g. logging / output filenames).
    model.geometry_encoder._swapped_from = type(old_encoder).__name__
    model.geometry_encoder._swap_spec = {
        "encoder_type": et,
        "model_path": model_path,
        "reference_frame": reference_frame,
    }

    if verbose:
        new_name = type(new_encoder).__name__
        feat_dim = new_encoder.get_feature_dim()
        ps = getattr(new_encoder, "patch_size", "?")
        print(f"[SwapEncoder] Done. geometry_encoder is now {new_name} "
              f"(feature_dim={feat_dim}, patch_size={ps}).")

    return True
