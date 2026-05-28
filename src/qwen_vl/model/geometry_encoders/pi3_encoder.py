"""Pi3 geometry encoder implementation.

Pi3 (https://github.com/InternRobotics/Pi3) is a multi-view 3D reconstruction
transformer with a DINOv2-Large encoder and a 36-block RoPE decoder. The
decoder concatenates the last two block outputs to give a `2 * dec_embed_dim`
feature vector per token (= 2048 for the `large` decoder), which matches
StreamVGGT's 2048-dim feature space exactly.

This wrapper exposes the same `encode(images) -> (features, concat_features)`
interface as the StreamVGGT / VGGT encoders so it can be dropped into
`Qwen2_5_VLForConditionalGenerationWithVGGT._process_geometry_features`
without modification:

    features        : (n_image, n_patches, 2048)
                      patch tokens only (used by `geometry_merger`).

    concat_features : (n_image, n_register + n_patches, 2048)
                      register tokens + patch tokens (used by
                      `concat_geometry_merger`). Plays the same role as
                      StreamVGGT's `[camera_token, patch_tokens]`.

The downstream `Concat_GeometryFeatureMerger` is shape-agnostic (just an MLP
over the last dim) and the trained `feature_fusion` uses cross-attention, so
we are free to expose Pi3's 5 register tokens without breaking anything.
"""

import torch
import torch.nn as nn
from typing import Optional

from .base import BaseGeometryEncoder, GeometryEncoderConfig


class Pi3Encoder(BaseGeometryEncoder):
    """Pi3 geometry encoder wrapper."""

    def __init__(self, config: GeometryEncoderConfig):
        super().__init__(config)

        print("Initializing Pi3 Encoder...")

        # Lazy import to avoid circular dependencies
        from ..pi3.models.pi3 import Pi3

        # Initialize Pi3 model (will be overwritten by load_model() if a
        # checkpoint path is supplied via geometry_encoder_path).
        self.pi3 = Pi3()

        if self.freeze_encoder:
            for param in self.pi3.parameters():
                param.requires_grad = False

        self.patch_size = 14

    def encode(self, images: torch.Tensor):
        """Encode images using Pi3.

        Args:
            images: ``(N, 3, H, W)`` (or already ``(B, N, 3, H, W)``) tensor
                of pixel values in [0, 1] (Pi3 normalises with ImageNet stats
                internally below).

        Returns:
            features        : ``(N, n_patches, 2*dec_embed_dim)``
            concat_features : ``(N, n_register + n_patches, 2*dec_embed_dim)``
        """
        self.pi3.eval()

        # Match precision conventions used by StreamVGGT/VGGT encoders.
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                # (N, C, H, W) -> (1, N, C, H, W) (Pi3 expects a leading batch dim)
                if images.dim() == 4:
                    images = images.unsqueeze(0)
                imgs = (images - self.pi3.image_mean) / self.pi3.image_std
                B, N, C, H, W = imgs.shape

                imgs = imgs.reshape(B * N, C, H, W)
                hidden = self.pi3.encoder(imgs, is_training=True)
                if isinstance(hidden, dict):
                    hidden = hidden["x_norm_patchtokens"]

                concat_features, _ = self.pi3.decode(hidden, N, H, W)
                # concat_features: (B*N, n_register + n_patches, 2 * dec_embed_dim)

                features = concat_features[:, self.pi3.patch_start_idx:, :]
                # features: (B*N, n_patches, 2 * dec_embed_dim)

                if B == 1:
                    features = features.reshape(N, *features.shape[1:])
                    concat_features = concat_features.reshape(N, *concat_features.shape[1:])

        return features, concat_features

    def get_feature_dim(self) -> int:
        """Get Pi3 feature dimension (decoder concatenates two layers)."""
        return self.pi3.dec_embed_dim * 2

    def forward(self, images: torch.Tensor):
        """Forward pass for compatibility."""
        return self.encode(images)

    def load_model(self, model_path: str) -> None:
        """Load pretrained Pi3 weights from a local dir or HF Hub id."""
        from ..pi3.models.pi3 import Pi3

        self.pi3 = Pi3.from_pretrained(model_path)

        if self.freeze_encoder:
            for param in self.pi3.parameters():
                param.requires_grad = False
