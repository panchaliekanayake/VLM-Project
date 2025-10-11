#CLIP version (with margin threshold + face-crop + denser sampling)

import os
import math
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# -----------------------
# Config
# -----------------------
VIDEO_PATH = "/home/labuser/Datasets/Videos/P1.avi" 
TARGET_SPACING_SEC = 2.5   # denser sampling for ~4 min video => ~96 frames
SAVE_FRAMES = False        # set True to export sampled (cropped) frames as images
OUTPUT_DIR = "/home/labuser/research/VLM-Project/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Decision margin: how confident we need to be to call it smile/not-smile
MARGIN_THRESHOLD = 0.01

# Face crop settings
FACE_CROP = True
FACE_PADDING = 0.20  # expand detected box by 20% on each side (clamped to image)

# Zero-shot prompts
TEXT_PROMPTS = ["a person smiling", "a person not smiling"]

# -----------------------
# Load CLIP
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# -----------------------
# Face detector (OpenCV Haar cascade)
# -----------------------
if FACE_CROP:
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_path)

def detect_and_crop_face(bgr_img):
    """Detect largest face; return cropped RGB array (or original if none)."""
    if not FACE_CROP:
        return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB), False
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    if len(faces) == 0:
        # No face -> fallback to full frame
        return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB), False

    # pick largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    H, W = bgr_img.shape[:2]

    # pad box
    pad_x = int(w * FACE_PADDING)
    pad_y = int(h * FACE_PADDING)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + w + pad_x)
    y2 = min(H, y + h + pad_y)

    crop = bgr_img[y1:y2, x1:x2]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), True

# -----------------------
# Helpers
# -----------------------
def time_spaced_indices(total_frames: int, fps: float, spacing_sec: float):
    """Return frame indices spaced by ~spacing_sec."""
    if total_frames <= 0 or fps <= 0:
        return []
    step = max(int(round(spacing_sec * fps)), 1)
    idxs = list(range(0, total_frames, step))
    if len(idxs) == 0:
        idxs = [0]
    return idxs

def analyze_video(video_path: str, spacing_sec: float, save_frames: bool):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        # fallback guess to avoid division by zero
        fps = 25.0

    # Build evenly time-spaced frame indices
    idxs = time_spaced_indices(total_frames, fps, spacing_sec)

    # Pre-encode prompts once
    with torch.no_grad():
        text_inputs = processor(text=TEXT_PROMPTS, images=None, return_tensors="pt", padding=True).to(device)
        text_embeds = model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    frame_save_dir = os.path.join(OUTPUT_DIR, "sampled_frames_clip")
    if save_frames:
        os.makedirs(frame_save_dir, exist_ok=True)

    results = []
    for fi in tqdm(idxs, desc="Evaluating frames (CLIP)"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue

        # Face-crop (fallback to full frame if no face)
        rgb_img, face_found = detect_and_crop_face(frame_bgr)
        pil_img = Image.fromarray(rgb_img)

        if save_frames:
            out_name = os.path.join(frame_save_dir, f"frame_{fi:06d}.jpg")
            pil_img.save(out_name, quality=95)

        # CLIP image -> features
        with torch.no_grad():
            image_inputs = processor(images=pil_img, return_tensors="pt").to(device)
            image_embeds = model.get_image_features(**image_inputs)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

            # Cosine similarity with prompts
            sim = image_embeds @ text_embeds.T  # shape [1, 2]
            sim_smile = sim[0, 0].item()
            sim_not  = sim[0, 1].item()
            margin = sim_smile - sim_not

            # Thresholded decision
            if margin >= MARGIN_THRESHOLD:
                label = "Yes"
            elif margin <= -MARGIN_THRESHOLD:
                label = "No"
            else:
                label = "Uncertain"

        ts = fi / fps
        results.append({
            "frame_index": fi,
            "timestamp_sec": ts,
            "face_detected": bool(face_found),
            "clip_sim_smiling": sim_smile,
            "clip_sim_not_smiling": sim_not,
            "smile_margin": margin,
            "smiled": label
        })

    cap.release()
    return pd.DataFrame(results), fps, total_frames

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    df, fps, total_frames = analyze_video(VIDEO_PATH, TARGET_SPACING_SEC, SAVE_FRAMES)

    if df.empty:
        raise RuntimeError("No frames analyzed — check the video path or codec.")

    # Majority decision across sampled frames (exclude "Uncertain")
    df_sorted = df.sort_values("frame_index")
    yes_count = (df_sorted["smiled"] == "Yes").sum()
    no_count = (df_sorted["smiled"] == "No").sum()
    uncertain_count = (df_sorted["smiled"] == "Uncertain").sum()
    total_votes = yes_count + no_count

    if total_votes == 0 or yes_count == no_count:
        majority_label = "Tie"
    else:
        majority_label = "Yes" if yes_count > no_count else "No"

    yes_pct = (yes_count / total_votes * 100.0) if total_votes > 0 else 0.0

    # Save CSV
    base = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    csv_path = os.path.join(OUTPUT_DIR, f"{base}_smile_majority_clip_dense_face_threshold.csv")
    df_sorted.to_csv(csv_path, index=False)

    # Print summary
    duration_sec = total_frames / fps if fps > 0 else float("nan")
    print("\n--- Summary (CLIP) ---")
    print(f"Video: {VIDEO_PATH}")
    print(f"FPS: {fps:.3f} | Total frames: {total_frames} | Duration: {duration_sec/60:.2f} min")
    print(f"Sampled frames: {len(df_sorted)} (every ~{TARGET_SPACING_SEC:.1f}s)")
    print(f"Per-frame smiles: Yes={yes_count}, No={no_count}, Uncertain={uncertain_count}, Yes% (of decided)={yes_pct:.1f}%")
    print(f"Final (majority) answer: {majority_label}")
    print(f"CSV saved: {csv_path}")
