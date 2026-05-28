"""Geometry encoders for 3D scene understanding."""

from .base import BaseGeometryEncoder, GeometryEncoderConfig
from .factory import create_geometry_encoder, get_available_encoders
from .vggt_encoder import VGGTEncoder
from .pi3_encoder import Pi3Encoder
from .cut3r_encoder import CUT3REncoder
from .noisy_patch import patch_geometry_encoder_with_noise, parse_noise_spec
from .swap_patch import swap_geometry_encoder

__all__ = [
    "BaseGeometryEncoder",
    "GeometryEncoderConfig",
    "create_geometry_encoder",
    "get_available_encoders",
    "VGGTEncoder",
    "Pi3Encoder",
    "CUT3REncoder",
    "patch_geometry_encoder_with_noise",
    "parse_noise_spec",
    "swap_geometry_encoder",
]
