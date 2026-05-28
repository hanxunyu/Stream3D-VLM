
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .base import BaseGeometryEncoder, GeometryEncoderConfig


# Default location of CUT3R/src in this repo (bundled via the VLM-3R
# reference clone). Override with CUT3R_REPO_PATH env var if you placed it
# elsewhere.
_DEFAULT_CUT3R_REPO = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "..",  # geometry_encoders/ -> model -> qwen_vl -> src -> repo
        "src", "sg3d_eval", "reference_repo", "VLM-3R", "CUT3R", "src",
    )
)


def _ensure_cut3r_path(extra_path: Optional[str] = None) -> str:
    repo_path = (
        extra_path
        or os.environ.get("CUT3R_REPO_PATH")
        or _DEFAULT_CUT3R_REPO
    )
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(
            f"[CUT3REncoder] CUT3R src repo not found at {repo_path!r}. "
            f"Either clone https://github.com/CUT3R/CUT3R and set the "
            f"CUT3R_REPO_PATH env var to its 'src' subdirectory, or pass "
            f"cut3r_repo_path=... in encoder_kwargs."
        )
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    return repo_path


class CUT3REncoder(BaseGeometryEncoder):
    """CUT3R geometry encoder wrapper.

    Optional ``encoder_kwargs`` (read from ``config.encoder_kwargs``):
        cut3r_repo_path (str)  : path to .../CUT3R/src (overrides default)
        internal_size  (tuple) : (H', W') multiple of 16, fed to CUT3R.
                                 Default (384, 512) -- close to Stream3D-LLM's
                                 (392, 518) and divisible by 16.
        feature_proj   (str)   : 'zero_pad' (default) | 'tile' | 'interp'
        target_patch_size (int): patch_size advertised to Stream3D-LLM.
                                 Default 14 (matches StreamVGGT).
    """

    NATIVE_FEATURE_DIM = 768           # CUT3R dec_embed_dim
    TARGET_FEATURE_DIM = 2048          # Stream3D-LLM merger context_dim
    DEFAULT_INTERNAL_SIZE = (384, 512)
    CUT3R_PATCH_SIZE = 16

    def __init__(self, config: GeometryEncoderConfig):
        super().__init__(config)

        kw = dict(config.encoder_kwargs or {})
        cut3r_repo = kw.get("cut3r_repo_path", None)
        _ensure_cut3r_path(cut3r_repo)

        # Lazy import after sys.path is patched.
        from dust3r.model import ARCroco3DStereo  # noqa: F401  (validate import)

        self.internal_size = tuple(kw.get("internal_size", self.DEFAULT_INTERNAL_SIZE))
        # feature_proj resolution order:
        #   encoder_kwargs['feature_proj']  >  env var  >  'zero_pad'
        self.feature_proj = str(
            kw.get(
                "feature_proj",
                os.environ.get("STREAM3D_CUT3R_FEATURE_PROJ", "zero_pad"),
            )
        )
        self.patch_size = int(kw.get("target_patch_size", 14))

        # CUT3R's architecture is encoded in the .pth, so we don't construct a
        # default network here -- it's instantiated inside load_model().
        self.cut3r = None

        if config.model_path:
            self.load_model(config.model_path)

    # ------------------------------------------------------------------ #
    # Public BaseGeometryEncoder API                                      #
    # ------------------------------------------------------------------ #

    def get_feature_dim(self) -> int:
        return self.TARGET_FEATURE_DIM

    def forward(self, images: torch.Tensor):
        return self.encode(images)

    def load_model(self, model_path: str) -> None:
        """Load CUT3R weights from a .pth file (or HF id)."""
        _ensure_cut3r_path()
        from dust3r.model import ARCroco3DStereo

        if not os.path.exists(model_path) and not model_path.startswith("hf://"):
            raise FileNotFoundError(
                f"[CUT3REncoder] checkpoint not found: {model_path!r}"
            )

        print(f"[CUT3REncoder] Loading CUT3R from {model_path}")
        net = ARCroco3DStereo.from_pretrained(model_path)
        net.eval()
        if self.freeze_encoder:
            for p in net.parameters():
                p.requires_grad = False

        # Assigning to self.cut3r registers it as a sub-module so .to(),
        # .state_dict(), .parameters() all see it.
        self.cut3r = net

    def encode(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode (n_image, 3, H, W) images in [0, 1] to geometry features.

        Returns:
            features        : (n_image, n_patches_target, 2048)
            concat_features : (n_image, 1 + n_patches_target, 2048)
                              first token along dim-1 is the projected pose
                              token (analogous to StreamVGGT's camera_token).
        """
        assert self.cut3r is not None, (
            "CUT3REncoder weights not loaded. Call load_model('/path/to/cut3r_*.pth')."
        )

        cut3r = self.cut3r
        cut3r.eval()
        device = images.device

        # ---- 1. Normalise to [-1, 1] (DUSt3R / CUT3R ImgNorm convention).
        x = images
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            x = x.float()
        x = x * 2.0 - 1.0

        # ---- 2. Internal resize to a CUT3R-friendly grid (multiple of 16).
        H_in, W_in = x.shape[-2:]
        H_int, W_int = self.internal_size
        if (H_in, W_in) != (H_int, W_int):
            x_in = F.interpolate(
                x, size=(H_int, W_int), mode="bilinear", align_corners=False
            )
        else:
            x_in = x

        # ---- 3. Build the per-frame views list.
        n_image = x_in.shape[0]
        views = []
        for i in range(n_image):
            img_i = x_in[i:i + 1]  # (1, 3, H_int, W_int)
            views.append({
                "img": img_i,
                "ray_map": torch.full(
                    (1, 6, H_int, W_int), float("nan"),
                    device=device, dtype=img_i.dtype,
                ),
                "true_shape": torch.tensor([[H_int, W_int]], device=device),
                "idx": i,
                "instance": [str(i)],
                "camera_pose": torch.eye(4, device=device).unsqueeze(0),
                "img_mask": torch.tensor([True], device=device),
                "ray_mask": torch.tensor([False], device=device),
                "update": torch.tensor([True], device=device),
                "reset": torch.tensor([False], device=device),
            })

        # ---- 4. CUT3R recurrent rollout (mirrors VLM-3R's Cut3rEncoder).
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16

        patch_features = []
        camera_tokens = []
        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda", dtype=amp_dtype
        ):
            shape, feat_ls, pos = cut3r._encode_views(views)
            feat = feat_ls[-1]
            state_feat, state_pos = cut3r._init_state(feat[0], pos[0])
            mem = cut3r.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
            init_state_feat = state_feat.clone()

            for i in range(len(views)):
                feat_i = feat[i].to(state_feat.dtype)
                pos_i = pos[i]
                if cut3r.pose_head_flag:
                    global_img_feat_i = cut3r._get_img_level_feat(feat_i)
                    if i == 0:
                        pose_feat_i = cut3r.pose_token.expand(feat_i.shape[0], -1, -1)
                    else:
                        pose_feat_i = cut3r.pose_retriever.inquire(
                            global_img_feat_i, mem
                        )
                    pose_pos_i = -torch.ones(
                        feat_i.shape[0], 1, 2,
                        device=feat_i.device, dtype=pos_i.dtype,
                    )
                else:
                    pose_feat_i = None
                    pose_pos_i = None

                new_state_feat, dec = cut3r._recurrent_rollout(
                    state_feat, state_pos, feat_i, pos_i,
                    pose_feat_i, pose_pos_i, init_state_feat,
                    img_mask=views[i]["img_mask"],
                    reset_mask=views[i]["reset"],
                    update=views[i].get("update", None),
                )

                if cut3r.pose_head_flag:
                    out_pose_feat_i = dec[-1][:, 0:1]
                    new_mem = cut3r.pose_retriever.update_mem(
                        mem, global_img_feat_i, out_pose_feat_i,
                    )
                else:
                    new_mem = mem

                # Update masks (here always True, but kept for fidelity).
                img_mask = views[i]["img_mask"]
                update = views[i].get("update", None)
                update_mask = (
                    img_mask & update if update is not None else img_mask
                )
                update_mask = update_mask[:, None, None].to(state_feat.dtype)
                state_feat = new_state_feat * update_mask + state_feat * (1 - update_mask)
                mem = new_mem * update_mask + mem * (1 - update_mask)

                # dec[-1]: (1, 1 + n_patches_cut3r, 768)
                camera_tokens.append(dec[-1][:, :1].clone())
                patch_features.append(dec[-1][:, 1:].clone())

        patch_features = torch.cat(patch_features, dim=0)   # (n_image, n_patches_cut3r, 768)
        camera_tokens = torch.cat(camera_tokens, dim=0)     # (n_image, 1, 768)

        # ---- 5. Bilinear resample patch features to Stream3D-LLM target grid.
        Hp_cut = H_int // self.CUT3R_PATCH_SIZE
        Wp_cut = W_int // self.CUT3R_PATCH_SIZE
        Hp_tgt = H_in // self.patch_size
        Wp_tgt = W_in // self.patch_size

        n_patches_cut = patch_features.shape[1]
        if n_patches_cut != Hp_cut * Wp_cut:
            raise RuntimeError(
                f"[CUT3REncoder] Unexpected CUT3R patch count {n_patches_cut} "
                f"vs Hp_cut*Wp_cut={Hp_cut * Wp_cut} for internal_size="
                f"{self.internal_size}. Pose token may have leaked in."
            )

        patch_grid = patch_features.reshape(
            n_image, Hp_cut, Wp_cut, -1,
        ).permute(0, 3, 1, 2)  # (n_image, 768, Hp_cut, Wp_cut)
        if (Hp_cut, Wp_cut) != (Hp_tgt, Wp_tgt):
            patch_grid = F.interpolate(
                patch_grid.float(),
                size=(Hp_tgt, Wp_tgt),
                mode="bilinear",
                align_corners=False,
            ).to(patch_features.dtype)
        patch_features = patch_grid.permute(0, 2, 3, 1).reshape(
            n_image, Hp_tgt * Wp_tgt, -1,
        )

        # ---- 6. Project 768 -> 2048.
        patch_features = self._project_to_target_dim(patch_features)
        camera_tokens = self._project_to_target_dim(camera_tokens)

        # ---- 7. Final concat_features = [pose, patches]; features = patches.
        features = patch_features
        concat_features = torch.cat([camera_tokens, patch_features], dim=1)

        return features, concat_features

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _project_to_target_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Project last dim from NATIVE (768) to TARGET (2048)."""
        in_dim = x.shape[-1]
        if in_dim == self.TARGET_FEATURE_DIM:
            return x
        if self.feature_proj == "zero_pad":
            pad = self.TARGET_FEATURE_DIM - in_dim
            if pad < 0:
                return x[..., : self.TARGET_FEATURE_DIM]
            zeros = x.new_zeros(*x.shape[:-1], pad)
            return torch.cat([x, zeros], dim=-1)
        if self.feature_proj == "tile":
            n_repeats = (self.TARGET_FEATURE_DIM + in_dim - 1) // in_dim
            tiled = x.repeat(*([1] * (x.dim() - 1)), n_repeats)
            return tiled[..., : self.TARGET_FEATURE_DIM]
        if self.feature_proj == "interp":
            orig_shape = x.shape
            x_flat = x.reshape(-1, 1, in_dim).float()
            x_proj = F.interpolate(
                x_flat,
                size=self.TARGET_FEATURE_DIM,
                mode="linear",
                align_corners=False,
            )
            return x_proj.reshape(*orig_shape[:-1], self.TARGET_FEATURE_DIM).to(x.dtype)
        raise ValueError(
            f"[CUT3REncoder] Unknown feature_proj={self.feature_proj!r}; "
            f"expected one of 'zero_pad' / 'tile' / 'interp'."
        )
