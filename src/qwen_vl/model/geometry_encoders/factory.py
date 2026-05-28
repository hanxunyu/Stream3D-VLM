"""Factory for creating geometry encoders."""

from typing import Optional
from .base import BaseGeometryEncoder, GeometryEncoderConfig
from .streamvggt_encoder import StreamVGGTEncoder
from .vggt_encoder import VGGTEncoder
from .pi3_encoder import Pi3Encoder
from .cut3r_encoder import CUT3REncoder


def create_geometry_encoder(
    encoder_type: str,
    model_path: Optional[str] = None,
    reference_frame: str = "first",
    freeze_encoder: bool = True,
    train_or_eval_mode: str = "eval",
    **encoder_kwargs
) -> BaseGeometryEncoder:
    """
    Factory function to create geometry encoders.

    Args:
        encoder_type: Type of encoder ("streamvggt", "vggt", "pi3", "cut3r").
        model_path: Path to pretrained model.
        reference_frame: Reference frame setting (only used by streamvggt / vggt).
        freeze_encoder: Whether to freeze encoder parameters.
        **encoder_kwargs: Additional encoder-specific arguments,
            forwarded via ``GeometryEncoderConfig.encoder_kwargs``.

    Returns:
        Geometry encoder instance.
    """
    config = GeometryEncoderConfig(
        encoder_type=encoder_type,
        model_path=model_path,
        reference_frame=reference_frame,
        freeze_encoder=freeze_encoder,
        train_or_eval_mode=train_or_eval_mode,
        encoder_kwargs=encoder_kwargs,
    )

    et = encoder_type.lower()
    if et == "streamvggt":
        return StreamVGGTEncoder(config)
    elif et == "vggt":
        return VGGTEncoder(config)
    elif et == "pi3":
        return Pi3Encoder(config)
    elif et == "cut3r":
        return CUT3REncoder(config)
    else:
        raise ValueError(f"Unknown geometry encoder type: {encoder_type}")


def get_available_encoders():
    """Get list of available encoder types."""
    return ["streamvggt", "vggt", "pi3", "cut3r"]
