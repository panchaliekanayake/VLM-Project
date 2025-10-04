
import os, re, numpy as np, pandas as pd, torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

# ---------------- Paths ----------------
TRANSCRIPTS_PATH = "/home/labuser/research/VLM-Project/data/Transcripts_All.xlsx"
SCORES_PATH      = "/home/labuser/research/VLM-Project/data/All_Scores.xlsx"
OUT_PATH         = "/home/labuser/research/VLM-Project/data/focused_with_calibration.csv"

# ---------------- Load data ----------------
if not os.path.exists(TRANSCRIPTS_PATH):
    raise FileNotFoundError(f"Missing: {TRANSCRIPTS_PATH}")
if not os.path.exists(SCORES_PATH):
    raise FileNotFoundError(f"Missing: {SCORES_PATH}")

transcripts_df = pd.read_excel(TRANSCRIPTS_PATH)
scores_df      = pd.read_excel(SCORES_PATH)

transcripts_df.columns = [c.strip() for c in transcripts_df.columns]
scores_df.columns      = [c.strip() for c in scores_df.columns]

def ensure_col(df, want, fallbacks=()):
    if want in df.columns: return want
    for fb in fallbacks:
        if fb in df.columns:
            df.rename(columns={fb: want}, inplace=True)
            return want
    raise KeyError(f"{want} not found; available: {list(df.columns)}")

ensure_col(transcripts_df, "Participant", ("participant","ID","Pid"))
ensure_col(transcripts_df, "Transcripts", ("Transcripts ","Transcript","Text","Interview","Transcript_Text"))
ensure_col(scores_df,      "Participant", ("participant","ID","Pid"))
if "Focused" not in scores_df.columns:
    ensure_col(scores_df, "Focused", ("focused","Focus","FOCUSED","focus_score"))

# ---------------- Model ----------------
device = 0 if torch.cuda.is_available() else -1
model_name = "microsoft/phi-3-mini-4k-instruct"  # primary
print("Loading model:", model_name)
try:
    tok = AutoTokenizer.from_pretrained(model_name)
    pad_id = getattr(tok, "pad_token_id", getattr(tok, "eos_token_id", None))
    generator = pipeline(
        "text-generation",
        model=model_name,
        device=device,
        # note: transformers warns torch_dtype is deprecated; safe to keep
        torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
        max_new_tokens=32,
        do_sample=False,
        top_p=1.0,
        pad_token_id=pad_id,
        return_full_text=False
    )
except Exception as e:
    print(f"Primary model failed ({model_name}): {e}")
    model_name = "google/flan-t5-small"
    print("Falling back to:", model_name)
    generator = pipeline("text2text-generation", model=model_name, device=device)

print(f"✅ Model used: {model_name}")

# ---------------- Prompt (2–7 scale) ----------------
BASE_PROMPT = (
    "You are an evaluator.\n"
    "Rate how **Focused** the Interviewee is on a 2–7 scale (decimals allowed, 5 = average focus).\n"
    "Definition:\n"
    "- 2–3: Mostly off-topic or disorganized.\n"
    "- 4–5: Generally answers but drifts sometimes.\n"
    "- 6–7: Consistently clear, direct, and organized.\n"
    "Judge only the Interviewee’s responses. Return only a number between 2 and 7.\n\n"
    "Transcript:\n{transcript}\n\nScore:"
)
PROMPT_VARIANTS = [
    BASE_PROMPT,
    BASE_PROMPT.replace("evaluator","strict evaluator"),
    BASE_PROMPT.replace("Rate","Give a numeric score for")
]

def extract_score(text: str):
    nums = re.findall(r'[-+]?\d*\.?\d+', str(text))
    for n in nums:
        try:
            v = float(n)
            if 2.0 <= v <= 7.0:
                return v
        except:
            pass
    return None

def run_generator(prompt: str):
    """Handle both causal and text2text pipeline return formats."""
    out = generator(prompt)
    if isinstance(out, list):
        if out and isinstance(out[0], dict) and "generated_text" in out[0]:
            return out[0]["generated_text"]
        # some text2text pipelines return [{"generated_text": "..."}] as well
        return str(out[0])
    return str(out)

def score_transcript(transcript: str):
    scores = []
    for p in PROMPT_VARIANTS:
        txt = run_generator(p.format(transcript=transcript))
        s = extract_score(txt)
        if s is not None:
            scores.append(s)
    # one strict retry if all failed
    if not scores:
        txt = run_generator(BASE_PROMPT.format(transcript=transcript) + " Return only the number.")
        s = extract_score(txt)
        if s is not None: scores.append(s)
    return float(np.mean(scores)) if scores else np.nan

# ---------------- Inference ----------------
print("\nScoring all participants...")
pred_rows = []
for _, row in tqdm(transcripts_df.iterrows(), total=len(transcripts_df)):
    part = row["Participant"]
    text = str(row["Transcripts"]).strip()
    if not text:
        pred_rows.append({"Participant": part, "Predicted_Focused": np.nan})
        continue
    try:
        score = score_transcript(text)
    except Exception as e:
        print(f"[Warn] {part}: {e}")
        score = np.nan
    pred_rows.append({"Participant": part, "Predicted_Focused": score})

pred_df = pd.DataFrame(pred_rows)
final_df = pred_df.merge(scores_df[["Participant","Focused"]], on="Participant", how="left")

# ---------------- Metrics helpers ----------------
def report(y_true, y_pred, tag):
    mask = y_true.notna() & y_pred.notna()
    if not mask.any():
        print(f"\n=== {tag} ===\nNo overlapping rows.")
        return mask
    r2  = r2_score(y_true[mask], y_pred[mask])
    mse = mean_squared_error(y_true[mask], y_pred[mask])
    rho = spearmanr(y_true[mask], y_pred[mask])[0] if mask.sum() >= 3 else np.nan
    print(f"\n=== {tag} ===")
    print(f"R²: {r2:.4f} | MSE: {mse:.4f} | Spearman ρ: {rho:.4f}")
    return mask

# ---------------- Raw metrics ----------------
y_true = pd.to_numeric(final_df["Focused"], errors="coerce")
y_pred = pd.to_numeric(final_df["Predicted_Focused"], errors="coerce")
mask   = report(y_true, y_pred, "Raw (uncalibrated)")

print("\n=== Sample (30 before calibration) ===")
print(final_df.sample(min(30,len(final_df)), random_state=42)
      [["Participant","Focused","Predicted_Focused"]]
      .to_string(index=False))

# ---------------- Linear calibration (safe with NaNs) ----------------
valid = y_true.notna() & y_pred.notna()
if valid.sum() >= 3 and np.nanstd(y_pred[valid]) > 1e-6:
    reg = LinearRegression()
    reg.fit(y_pred[valid].values.reshape(-1,1), y_true[valid].values)

    # Predict ONLY for valid rows, keep NaN elsewhere
    y_cal = np.full(len(y_pred), np.nan, dtype=float)
    y_cal[valid] = reg.predict(y_pred[valid].values.reshape(-1,1))

    # Clip to your judges' range (2–7)
    y_cal = np.clip(y_cal, 2.0, 7.0)
    final_df["Calibrated_Focused"] = y_cal

    # Report using the same 'valid' mask
    _ = report(y_true, final_df["Calibrated_Focused"], "After Linear Calibration")
else:
    final_df["Calibrated_Focused"] = np.nan
    print("\n[Note] Not enough valid rows/variance for calibration; skipped.")

print("\n=== Sample (30 after calibration) ===")
print(final_df.sample(min(30,len(final_df)), random_state=123)
      [["Participant","Focused","Predicted_Focused","Calibrated_Focused"]]
      .to_string(index=False))

# ---------------- Save ----------------
final_df.to_csv(OUT_PATH, index=False)
print(f"\n💾 Saved results to: {OUT_PATH}")
