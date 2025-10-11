import os
import math
import cv2
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# -----------------------
# Config
# -----------------------
VIDEO_PATH = "/home/labuser/Datasets/Videos/P1.avi"  # <- change if needed
TARGET_SPACING_SEC = 5.0   # sample every N seconds; 5 sec => ~48 frames for a 4-min video
SAVE_FRAMES = False        # set True to export sampled frames as images
OUTPUT_DIR = "/home/labuser/research/VLM-Project/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        # fallback guess to avoid division by zero; most videos are >= 24 fps
        fps = 25.0

    # Build evenly time-spaced frame indices
    idxs = time_spaced_indices(total_frames, fps, spacing_sec)

    # Pre-encode prompts once
    with torch.no_grad():
        text_inputs = processor(text=TEXT_PROMPTS, images=None, return_tensors="pt", padding=True).to(device)
        text_embeds = model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    frame_save_dir = os.path.join(OUTPUT_DIR, "sampled_frames")
    if save_frames:
        os.makedirs(frame_save_dir, exist_ok=True)

    results = []
    for fi in tqdm(idxs, desc="Evaluating frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue

        # Convert to PIL RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        if save_frames:
            pil_img.save(os.path.join(frame_save_dir, f"frame_{fi:06d}.jpg"), quality=95)

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
            label = "Yes" if sim_smile >= sim_not else "No"

        ts = fi / fps
        results.append({
            "frame_index": fi,
            "timestamp_sec": ts,
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

    # Majority decision across sampled frames
    df_sorted = df.sort_values("frame_index")
    yes_count = (df_sorted["smiled"] == "Yes").sum()
    no_count = (df_sorted["smiled"] == "No").sum()
    total_votes = yes_count + no_count
    majority_label = "Yes" if yes_count > no_count else ("No" if no_count > yes_count else "Tie")
    yes_pct = (yes_count / total_votes * 100.0) if total_votes > 0 else 0.0

    # Save CSV
    base = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    csv_path = os.path.join(OUTPUT_DIR, f"{base}_smile_majority_clip.csv")
    df_sorted.to_csv(csv_path, index=False)

    # Print summary
    duration_sec = total_frames / fps if fps > 0 else float("nan")
    print("\n--- Summary ---")
    print(f"Video: {VIDEO_PATH}")
    print(f"FPS: {fps:.3f} | Total frames: {total_frames} | Duration: {duration_sec/60:.2f} min")
    print(f"Sampled frames: {len(df_sorted)} (every ~{TARGET_SPACING_SEC:.1f}s)")
    print(f"Per-frame smiles: Yes={yes_count}, No={no_count}, Yes%={yes_pct:.1f}%")
    print(f"Final (majority) answer: {majority_label}")
    print(f"\nPer-frame results saved to: {csv_path}")
