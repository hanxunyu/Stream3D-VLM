"""StreamVGGT geometry encoder implementation."""

import torch
import torch.nn as nn
from typing import Optional

from .base import BaseGeometryEncoder, GeometryEncoderConfig
from ..streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
from ..streamvggt.utils.geometry import unproject_depth_map_to_point_map

class StreamVGGTEncoder(BaseGeometryEncoder):
    """StreamVGGT geometry encoder wrapper."""
    
    def __init__(self, config: GeometryEncoderConfig):
        super().__init__(config)
        
        # Lazy import to avoid circular dependencies
        from ..streamvggt.models.streamvggt import StreamVGGT

        # Initialize StreamVGGT model
        self.streamvggt = StreamVGGT(enable_camera=True, enable_point=True, enable_depth=True, enable_track=False)
        
        # Freeze parameters if required
        if self.freeze_encoder:
            for param in self.streamvggt.parameters():
                param.requires_grad = False

        self.reference_frame = config.reference_frame
        self.train_or_eval_mode = config.train_or_eval_mode
        self.patch_size = 14
        
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using StreamVGGT."""
        self.streamvggt.eval()

        # Apply reference frame transformation
        images = self._apply_reference_frame_transform(images)
        # print(f"StreamVGGTEncoder: input images shape {images.shape}")
        # Determine dtype for mixed precision
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                # Get aggregated tokens from StreamVGGT
                images = images[None]
                model = self.streamvggt
                # print(f"self.train_or_eval_mode: {self.train_or_eval_mode}")
                if self.train_or_eval_mode == "train":
                    # print(f"StreamVGGTEncoder: set StreamVGGT to train mode")
                    aggregated_tokens_list, patch_start_idx = model.aggregator(images)
                    # print(f"StreamVGGTEncoder: the shape of images[None] is {images.shape}")
                    # print(f"StreamVGGTEncoder: aggregated tokens list[-2] shape {aggregated_tokens_list[-2].shape}")
                    features = aggregated_tokens_list[-2][0, :, patch_start_idx:]
                    # print(f"StreamVGGTEncoder: features shape {features.shape}")
                    
                    camera_token = aggregated_tokens_list[-2][0, :, 0:1]
                    # print(f"StreamVGGTEncoder: camera_token shape {camera_token.shape}")

                    concat_features = torch.cat([camera_token, features], dim=1)
                    # print(f"StreamVGGTEncoder: concat_features shape {concat_features.shape}")

                # # Predict Cameras
                # pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                # print(f"StreamVGGTEncoder: pose_enc shape {pose_enc.shape}")
                # # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
                # extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
                # print(f"StreamVGGTEncoder: extrinsic shape {extrinsic.shape}, intrinsic shape {intrinsic.shape}")

                # # Predict Depth Maps
                # depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, patch_start_idx)
                # print(f"StreamVGGTEncoder: depth_map shape {depth_map.shape}, depth_conf shape {depth_conf.shape}")  
                # # Predict Point Maps
                # point_map, point_conf = model.point_head(aggregated_tokens_list, images, patch_start_idx)
                # print(f"StreamVGGTEncoder: point_map shape {point_map.shape}, point_conf shape {point_conf.shape}")
                    
                # # Construct 3D Points from Depth Maps and Cameras
                # # which usually leads to more accurate 3D points than point map branch
                # point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                #                                                             extrinsic.squeeze(0), 
                #                                                             intrinsic.squeeze(0))
                # print(f"StreamVGGTEncoder: point_map_by_unprojection shape {point_map_by_unprojection.shape}")

            # Apply inverse reference frame transformation
            features = self._apply_inverse_reference_frame_transform(features)

        return features, concat_features
    
    def get_feature_dim(self) -> int:
        """Get StreamVGGT feature dimension."""
        return 2048  # StreamVGGT feature dimension

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass for compatibility."""
        return self.encode(images)
    
    def _apply_reference_frame_transform(self, images: torch.Tensor) -> torch.Tensor:
        """Apply reference frame transformation if needed."""
        if self.reference_frame != "first":
            return torch.flip(images, dims=(0,))
        return images
    
    def _apply_inverse_reference_frame_transform(self, features: torch.Tensor) -> torch.Tensor:
        """Apply inverse reference frame transformation if needed."""
        if self.reference_frame != "first":
            return torch.flip(features, dims=(0,))
        return features

    
    def load_model(self, model_path: str) -> None:
        """Load pretrained StreamVGGT model."""
        from ..streamvggt.models.streamvggt import StreamVGGT
        self.streamvggt = StreamVGGT.from_pretrained(model_path, enable_camera=True, enable_point=True, enable_depth=True, enable_track=False)

        # Freeze parameters if required
        if self.freeze_encoder:
            for param in self.streamvggt.parameters():
                param.requires_grad = False
