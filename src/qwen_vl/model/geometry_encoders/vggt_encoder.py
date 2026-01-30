"""VGGT geometry encoder implementation."""

import torch
import torch.nn as nn
from typing import Optional

from .base import BaseGeometryEncoder, GeometryEncoderConfig
from ..vggt.utils.pose_enc import pose_encoding_to_extri_intri
from ..vggt.utils.geometry import unproject_depth_map_to_point_map

class VGGTEncoder(BaseGeometryEncoder):
    """VGGT geometry encoder wrapper."""
    
    def __init__(self, config: GeometryEncoderConfig):
        super().__init__(config)
        
        # Lazy import to avoid circular dependencies
        from ..vggt.models.vggt import VGGT

        # Initialize VGGT model
        self.vggt = VGGT(enable_camera=True, enable_point=True, enable_depth=True, enable_track=False)
        
        # Freeze parameters if required
        if self.freeze_encoder:
            for param in self.vggt.parameters():
                param.requires_grad = False

        self.reference_frame = config.reference_frame    
        self.patch_size = 14
        
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using VGGT."""
        self.vggt.eval()

        # Apply reference frame transformation
        images = self._apply_reference_frame_transform(images)
        print(f"VGGTEncoder: input images shape {images.shape}")
        # Determine dtype for mixed precision
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                # Get aggregated tokens from VGGT
                images = images[None]
                model = self.vggt

                aggregated_tokens_list, patch_start_idx = model.aggregator(images)
                print(f"VGGTEncoder: the shape of images[None] is {images.shape}")
                print(f"VGGTEncoder: aggregated tokens list[-2] shape {aggregated_tokens_list[-2].shape}")
                features = aggregated_tokens_list[-2][0, :, patch_start_idx:]
                print(f"VGGTEncoder: features shape {features.shape}")
                
                camera_token = aggregated_tokens_list[-2][0, :, 0:1]
                print(f"VGGTEncoder: camera_token shape {camera_token.shape}")
                
                concat_features = torch.cat([camera_token, features], dim=1)
                print(f"VGGTEncoder: concat_features shape {concat_features.shape}")

                # # Predict Cameras
                # pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                # print(f"VGGTEncoder: pose_enc shape {pose_enc.shape}")
                # # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
                # extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
                # print(f"VGGTEncoder: extrinsic shape {extrinsic.shape}, intrinsic shape {intrinsic.shape}")

                # # Predict Depth Maps
                # depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, patch_start_idx)
                # print(f"VGGTEncoder: depth_map shape {depth_map.shape}, depth_conf shape {depth_conf.shape}")  
                # # Predict Point Maps
                # point_map, point_conf = model.point_head(aggregated_tokens_list, images, patch_start_idx)
                # print(f"VGGTEncoder: point_map shape {point_map.shape}, point_conf shape {point_conf.shape}")
                    
                # # Construct 3D Points from Depth Maps and Cameras
                # # which usually leads to more accurate 3D points than point map branch
                # point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                #                                                             extrinsic.squeeze(0), 
                #                                                             intrinsic.squeeze(0))
                # print(f"VGGTEncoder: point_map_by_unprojection shape {point_map_by_unprojection.shape}")

            # Apply inverse reference frame transformation
            features = self._apply_inverse_reference_frame_transform(features)

        return features, concat_features
    
    def get_feature_dim(self) -> int:
        """Get VGGT feature dimension."""
        return 2048  # VGGT feature dimension
    
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
        """Load pretrained VGGT model."""
        from ..vggt.models.vggt import VGGT
        self.vggt = VGGT.from_pretrained(model_path, enable_camera=True, enable_point=True, enable_depth=True, enable_track=False)
                
        # Freeze parameters if required
        if self.freeze_encoder:
            for param in self.vggt.parameters():
                param.requires_grad = False
