"""
Incremental 3D Reconstruction using StreamVGGT.
Generates step-by-step GLB point cloud files for visualization.

Usage:
    python run_incremental_reconstruction.py \
        --input /path/to/scene/color \
        --output /path/to/output \
        --ckpt /path/to/checkpoints.pth \
        --step 30 \
        --incremental_step 1 \
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
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# --- StreamVGGT dependencies ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STREAMVGGT_DIR = os.path.join(SCRIPT_DIR, "StreamVGGT")
sys.path.append(os.path.join(STREAMVGGT_DIR, "src"))

from visual_util import predictions_to_glb
from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.load_fn import load_and_preprocess_images
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri


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


def save_incremental_glbs(predictions, target_dir, conf_thres, incremental_step, mode, scannet_fix):
    """Save incremental GLB files at each step."""
    save_dir = os.path.join(target_dir, "incremental_steps")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    total_frames = predictions["images"].shape[0]
    step = incremental_step

    print(f"\n[Incremental] Generating step-by-step GLBs into '{save_dir}'")
    print(f"[Incremental] Total frames: {total_frames}, Step size: {step}")

    indices = list(range(step, total_frames, step))
    if indices[-1] != total_frames:
        indices.append(total_frames)

    for i in tqdm(indices, desc="Saving Incremental GLBs"):
        partial_preds = {}
        for key, value in predictions.items():
            if isinstance(value, (np.ndarray, torch.Tensor)) and value.shape[0] == total_frames:
                partial_preds[key] = value[:i]
            else:
                partial_preds[key] = value

        glbscene = predictions_to_glb(
            partial_preds,
            conf_thres=conf_thres,
            filter_by_frames="All",
            mask_black_bg=False,
            mask_white_bg=False,
            show_cam=True,
            mask_sky=False,
            target_dir=target_dir,
            prediction_mode=mode,
        )

        filename = f"step_{i:04d}.glb"
        glbscene.export(file_obj=os.path.join(save_dir, filename))

    print(f"[Incremental] Saved {len(indices)} GLB files.\n")


def main():
    parser = argparse.ArgumentParser(description="Incremental 3D Reconstruction with StreamVGGT (GLB output)")
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
    parser.add_argument("--incremental_step", type=int, default=1, help="Save a GLB every N frames (0 = final only)")

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

    # Save incremental GLBs
    if args.incremental_step > 0:
        save_incremental_glbs(predictions, target_dir, args.conf_thres, args.incremental_step, args.mode, args.scannet_fix)

    # Save final GLB
    print(f"Generating final GLB (conf_thres={args.conf_thres}%)...")
    glb_path = os.path.join(target_dir, f"scene_final_conf{args.conf_thres}.glb")
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=args.conf_thres,
        filter_by_frames="All",
        mask_black_bg=False,
        mask_white_bg=False,
        show_cam=True,
        mask_sky=False,
        target_dir=target_dir,
        prediction_mode=args.mode,
    )
    glbscene.export(file_obj=glb_path)

    print(f"\n[Done] Final GLB: {glb_path}")
    if args.incremental_step > 0:
        print(f"[Done] Incremental GLBs: {os.path.join(target_dir, 'incremental_steps')}/")


if __name__ == "__main__":
    main()
