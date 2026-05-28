"""
3D Reconstruction using StreamVGGT.
Generates a final RRD (Rerun) file for visualization.

Usage:
    python run_incremental_reconstruction_rrd.py \
        --input /path/to/scene/color \
        --output /path/to/output \
        --ckpt /path/to/checkpoints.pth \
        --step 30 \
        --conf_thres 50.0 \
        --scannet_fix
"""

import os
import sys
import glob
import re
import cv2
import torch
import numpy as np
import argparse
import shutil
import uuid
from huggingface_hub import hf_hub_download

# --- StreamVGGT dependencies ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STREAMVGGT_DIR = os.path.join(SCRIPT_DIR, "StreamVGGT")
sys.path.append(os.path.join(STREAMVGGT_DIR, "src"))

from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.load_fn import load_and_preprocess_images
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri


# =============================================================================
# predictions_to_rrd: self-contained RRD generation (no StreamVGGT modification)
# =============================================================================
def predictions_to_rrd(
    predictions,
    output_path="output.rrd",
    conf_thres=50.0,
    filter_by_frames="All",
    mask_black_bg=False,
    mask_white_bg=False,
    mask_sky=False,
    target_dir=None,
    prediction_mode="Depthmap and Camera Branch",
):
    """Convert predictions dict to a Rerun .rrd file."""
    if os.path.exists(output_path):
        print(f"Removing existing RRD file at {output_path}")
        os.remove(output_path)

    try:
        import rerun as rr
    except ImportError:
        print("Error: rerun-sdk is not installed. Please install it via 'pip install rerun-sdk'")
        return

    print(f"Rerun version: {getattr(rr, '__version__', 'Unknown')}")
    print(f"Building RRD recording -> {output_path}")

    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    rr.init("VGGT_Visualization", spawn=False, recording_id=str(uuid.uuid4()))

    # --- Data Extraction ---
    selected_frame_idx = None
    if filter_by_frames != "all" and filter_by_frames != "All":
        try:
            selected_frame_idx = int(filter_by_frames.split(":")[0])
        except (ValueError, IndexError):
            pass

    if "Pointmap" in prediction_mode:
        if "world_points" in predictions:
            pred_world_points = predictions["world_points"]
            pred_world_points_conf = predictions.get("world_points_conf", np.ones_like(pred_world_points[..., 0]))
        else:
            pred_world_points = predictions["world_points_from_depth"]
            pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))
    else:
        pred_world_points = predictions["world_points_from_depth"]
        pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))

    images = predictions["images"]
    camera_matrices = predictions["extrinsic"]
    intrinsic_matrices = predictions.get("intrinsic")

    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
        images = images[selected_frame_idx][None]
        camera_matrices = camera_matrices[selected_frame_idx][None]
        if intrinsic_matrices is not None:
            intrinsic_matrices = intrinsic_matrices[selected_frame_idx][None]

    num_frames = len(images)

    all_conf = pred_world_points_conf.reshape(-1)
    if conf_thres == 0.0:
        conf_threshold = 0.0
    else:
        conf_threshold = np.percentile(all_conf, conf_thres)

    # --- Rerun Logging Loop ---
    for i in range(num_frames):
        if hasattr(rr, "set_time_sequence"):
            rr.set_time_sequence("frame_idx", i)
        else:
            try:
                rr.set_time(sequence=i, timeline="frame_idx")
            except AttributeError:
                if i == 0:
                    print("WARNING: Neither 'set_time_sequence' nor 'set_time' found in rerun module.")

        # Camera pose
        w2c = np.eye(4)
        w2c[:3, :4] = camera_matrices[i]
        c2w = np.linalg.inv(w2c)
        translation = c2w[:3, 3]
        rotation_matrix = c2w[:3, :3]

        # Image dimensions
        if images.shape[1] == 3:
            H, W = images.shape[2], images.shape[3]
        else:
            H, W = images.shape[1], images.shape[2]

        # Intrinsics
        if intrinsic_matrices is not None:
            k_matrix = intrinsic_matrices[i]
        else:
            if i == 0:
                print("Warning: Using estimated intrinsics.")
            f_len = W * 0.8
            k_matrix = np.array([[f_len, 0, W / 2], [0, f_len, H / 2], [0, 0, 1]])

        # Image
        img_rgb = images[i]
        if img_rgb.ndim == 3 and img_rgb.shape[0] == 3:
            img_rgb = np.transpose(img_rgb, (1, 2, 0))
        img_uint8 = (img_rgb * 255).astype(np.uint8)

        # Log camera
        traj_path = f"world/cameras/{i}"
        rr.log(
            traj_path,
            rr.Transform3D(
                translation=translation,
                mat3x3=rotation_matrix,
            ),
            rr.Pinhole(
                image_from_camera=k_matrix,
                width=W,
                height=H,
                image_plane_distance=0.1,
                line_width=0.003,
                color=[0, 255, 255],
            ),
            rr.TransformAxes3D(
                axis_length=1e-5,
            ),
        )

        # Log point cloud
        points = pred_world_points[i].reshape(-1, 3)
        colors = img_uint8.reshape(-1, 3)
        conf = pred_world_points_conf[i].reshape(-1)

        mask = (conf >= conf_threshold) & (conf > 1e-5)
        if mask_black_bg:
            black_mask = colors.sum(axis=1) >= 16
            mask = mask & black_mask
        if mask_white_bg:
            white_mask = ~((colors[:, 0] > 240) & (colors[:, 1] > 240) & (colors[:, 2] > 240))
            mask = mask & white_mask

        points_filtered = points[mask]
        colors_filtered = colors[mask]

        if len(points_filtered) > 0:
            rr.log(
                f"world/points/{i}",
                rr.Points3D(points_filtered, colors=colors_filtered, radii=0.005),
            )

    rr.save(output_path)
    print(f"RRD file saved successfully: {output_path}")


# =============================================================================
# Shared utilities (same as GLB version)
# =============================================================================
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def load_model(ckpt_path="ckpt/checkpoints.pth"):
    print("Initializing StreamVGGT model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(ckpt_path):
        print(f"Local checkpoint not found at {ckpt_path}, downloading from Hugging Face...")
        try:
            ckpt_path = hf_hub_download(
                repo_id="lch01/StreamVGGT",
                filename="checkpoints.pth",
                revision="main",
                force_download=False
            )
        except Exception as e:
            print(f"Error downloading model: {e}")
            sys.exit(1)

    print(f"Loading checkpoint from {ckpt_path}")
    model = StreamVGGT()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model = model.to(device)
    model.eval()
    return model


def prepare_input(input_path, output_dir, step=1, max_frames=50):
    images_dir = os.path.join(output_dir, "images")
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    os.makedirs(images_dir)

    print(f"Processing input: {input_path}")

    if os.path.isdir(input_path):
        image_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.PNG']
        files = []
        for ext in image_exts:
            files.extend(glob.glob(os.path.join(input_path, ext)))
        files = sorted(files, key=natural_sort_key)

        total_files = len(files)
        if total_files == 0:
            raise ValueError(f"No images found in {input_path}")

        if step > 1:
            print(f"Applying stride: selecting every {step}-th image.")
            files = files[::step]

        if len(files) > max_frames:
            print(f"Limiting to first {max_frames} images.")
        files = files[:max_frames]
        print(f"Found {total_files} images. Using {len(files)} images for processing.")

        for idx, src_file in enumerate(files):
            ext = os.path.splitext(src_file)[1]
            dst_name = f"{idx:06d}{ext}"
            shutil.copy(src_file, os.path.join(images_dir, dst_name))

    elif os.path.isfile(input_path):
        print(f"Extracting frames from video...")
        vs = cv2.VideoCapture(input_path)
        video_fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / 1.0))

        count = 0
        saved_count = 0
        while True:
            ret, frame = vs.read()
            if not ret:
                break
            if count % frame_interval == 0:
                save_path = os.path.join(images_dir, f"{saved_count:06d}.png")
                cv2.imwrite(save_path, frame)
                saved_count += 1
            count += 1
        vs.release()
        print(f"Extracted {saved_count} frames from video.")
    else:
        raise ValueError("Invalid input path")

    return output_dir


def clean_scannet_borders(predictions, margin=10):
    """Remove black borders from ScanNet images by invalidating edge pixels."""
    if margin <= 0:
        return predictions

    print(f"Cleaning borders: margin={margin}px...")

    imgs = predictions["images"]
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.cpu().numpy()
    if imgs.shape[1] == 3 and imgs.ndim == 4:
        imgs = imgs.transpose(0, 2, 3, 1)

    s, h, w, c = imgs.shape

    target_keys = ["world_points_conf", "depth_conf"]
    for key in target_keys:
        if key in predictions:
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy()

    for i in range(s):
        mask = np.full((h, w), 255, dtype=np.uint8)
        mask[:margin, :] = 0
        mask[-margin:, :] = 0
        mask[:, :margin] = 0
        mask[:, -margin:] = 0

        valid_mask = (mask > 127)
        for key in target_keys:
            if key in predictions:
                current_conf = predictions[key][i]
                if current_conf.shape != valid_mask.shape:
                    predictions[key][i] = np.where(valid_mask[..., None], current_conf, -100.0)
                else:
                    predictions[key][i] = np.where(valid_mask, current_conf, -100.0)

    print("Border cleaning applied.")
    return predictions


def run_inference(target_dir, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_names = sorted(glob.glob(os.path.join(target_dir, "images", "*")))
    if len(image_names) == 0:
        raise ValueError("No images found for inference.")

    print(f"Preprocessing {len(image_names)} images...")
    images = load_and_preprocess_images(image_names).to(device)

    frames = []
    for i in range(images.shape[0]):
        frames.append({"img": images[i].unsqueeze(0)})

    print("Running StreamVGGT inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            output = model.inference(frames)

    results = {"pts3d": [], "conf": [], "depth": [], "depth_conf": [], "camera_pose": []}
    for res in output.ress:
        results["pts3d"].append(res['pts3d_in_other_view'].squeeze(0))
        results["conf"].append(res['conf'].squeeze(0))
        results["depth"].append(res['depth'].squeeze(0))
        results["depth_conf"].append(res['depth_conf'].squeeze(0))
        results["camera_pose"].append(res['camera_pose'].squeeze(0))

    predictions = {}
    predictions["images"] = images
    predictions["world_points"] = torch.stack(results["pts3d"], dim=0)
    predictions["world_points_conf"] = torch.stack(results["conf"], dim=0)
    predictions["depth"] = torch.stack(results["depth"], dim=0)
    predictions["depth_conf"] = torch.stack(results["depth_conf"], dim=0)
    predictions["pose_enc"] = torch.stack(results["camera_pose"], dim=0)

    print("Computing camera matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"].unsqueeze(0) if predictions["pose_enc"].ndim == 2 else predictions["pose_enc"],
        images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic.squeeze(0)
    predictions["intrinsic"] = intrinsic.squeeze(0) if intrinsic is not None else None

    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy()

    predictions["world_points_from_depth"] = predictions["world_points"]
    torch.cuda.empty_cache()
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Incremental 3D Reconstruction with StreamVGGT (RRD output)")
    parser.add_argument("--input", type=str, required=True, help="Image folder or video file path")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--npz_path", type=str, default=None, help="Directory to save/load predictions.npz (defaults to --output)")
    parser.add_argument("--ckpt", type=str, default="ckpt/checkpoints.pth", help="Model checkpoint path")
    parser.add_argument("--conf_thres", type=float, default=50.0, help="Confidence threshold (0-100)")
    parser.add_argument("--mode", type=str, default="Depthmap and Camera Branch",
                        choices=["Depthmap and Camera Branch", "Pointmap Branch"])
    parser.add_argument("--step", type=int, default=1, help="Frame sampling stride for input images")
    parser.add_argument("--max_frames", type=int, default=50, help="Maximum number of frames to process")
    parser.add_argument("--scannet_fix", action="store_true", help="Enable ScanNet black border removal")

    args = parser.parse_args()

    npz_dir = args.npz_path if args.npz_path else args.output
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)

    target_dir = args.output
    npz_path = os.path.join(npz_dir, "predictions.npz")

    # Load or run inference
    if os.path.exists(npz_path):
        print(f"Found existing predictions at {npz_path}, loading...")
        predictions = dict(np.load(npz_path, allow_pickle=True))
    else:
        target_dir = prepare_input(args.input, args.output, step=args.step, max_frames=args.max_frames)
        model = load_model(args.ckpt)
        predictions = run_inference(target_dir, model)
        print(f"Saving predictions to {npz_path}")
        np.savez(npz_path, **predictions)

    # ScanNet border fix
    if args.scannet_fix:
        predictions = clean_scannet_borders(predictions, margin=10)

    # Save final RRD
    print(f"Generating final RRD (conf_thres={args.conf_thres}%)...")
    rrd_path = os.path.join(target_dir, f"scene_final_conf{args.conf_thres}.rrd")
    predictions_to_rrd(
        predictions,
        output_path=rrd_path,
        conf_thres=args.conf_thres,
        filter_by_frames="All",
        mask_black_bg=False,
        mask_white_bg=False,
        mask_sky=False,
        target_dir=target_dir,
        prediction_mode=args.mode,
    )

    print(f"\n[Done] Final RRD: {rrd_path}")

if __name__ == "__main__":
    main()
