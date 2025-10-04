import os, re, json, numpy as np, pandas as pd, torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

# ---------------- Paths ----------------
TRANSCRIPTS_PATH = "/home/labuser/research/VLM-Project/data/Transcripts_All.xlsx"
SCORES_PATH      = "/home/labuser/research/VLM-Project/data/All_Scores.xlsx"
OUT_PATH         = "/home/labuser/research/VLM-Project/data/focused_with_calibration.csv"
METRICS_PATH     = OUT_PATH.replace(".csv", "_metrics.json")

# ---------------- Small controls ----------------
MAX_CHARS = 6000          # keep prompts lean
CLEAN_TEXT = True         # apply light cleaner
SEED = 42                 # determinism for generation

torch.manual_seed(SEED)
np.random.seed(SEED)

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

# ---------------- Tiny cleaner ----------------
FILLERS_RE = re.compile(r"\b(uh+|um+|erm+|ah+|mhm+|mm+|hmm+|like)\b[,.\s]*", re.I)
def clean_transcript(t: str) -> str:
    if not CLEAN_TEXT: return t
    t = re.sub(r"(?im)^Interviewer:\s.*$", "", t)   # drop interviewer lines
    t = t.replace("|", "\n")
    t = FILLERS_RE.sub("", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

# ---------------- Model (try a few) ----------------
device = 0 if torch.cuda.is_available() else -1
CANDIDATES = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta",
    "microsoft/phi-3-mini-4k-instruct",
    "google/flan-t5-small",   # last-resort
]

def load_generator(name):
    try:
        tok = AutoTokenizer.from_pretrained(name)
        pad_id = getattr(tok, "pad_token_id", getattr(tok, "eos_token_id", None))
        task = "text-generation" if "t5" not in name.lower() else "text2text-generation"
        gen = pipeline(
            task,
            model=name,
            device=device,
            dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
            max_new_tokens=32,
            do_sample=False,          # deterministic
            top_p=1.0,
            pad_token_id=pad_id,
            return_full_text=False,
        )
        return gen
    except Exception as e:
        print(f"[Skip] {name}: {e}")
        return None

generator = None
used_model = None
for m in CANDIDATES:
    print("Loading model:", m)
    generator = load_generator(m)
    if generator is not None:
        used_model = m
        break
if generator is None:
    raise RuntimeError("No candidate model could be loaded.")
print(f"✅ Model used: {used_model}")

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
    BASE_PROMPT.replace("Rate","Give a numeric score for"),
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
    out = generator(prompt)
    if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
        return out[0]["generated_text"]
    return str(out)

def score_transcript(transcript: str):
    t = clean_transcript(transcript)[:MAX_CHARS]
    scores = []
    for p in PROMPT_VARIANTS:
        txt = run_generator(p.format(transcript=t))
        s = extract_score(txt)
        if s is not None:
            scores.append(s)
    if not scores:  # one strict retry
        txt = run_generator(BASE_PROMPT.format(transcript=t) + " Return only the number.")
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
        return mask, None
    r2  = r2_score(y_true[mask], y_pred[mask])
    mse = mean_squared_error(y_true[mask], y_pred[mask])
    rho = spearmanr(y_true[mask], y_pred[mask])[0] if mask.sum() >= 3 else np.nan
    print(f"\n=== {tag} ===")
    print(f"R²: {r2:.4f} | MSE: {mse:.4f} | Spearman ρ: {rho:.4f}")
    return mask, {"r2": float(r2), "mse": float(mse), "spearman": float(rho) if rho==rho else None}

# ---------------- Raw metrics ----------------
y_true = pd.to_numeric(final_df["Focused"], errors="coerce")
y_pred = pd.to_numeric(final_df["Predicted_Focused"], errors="coerce")
mask, raw_stats = report(y_true, y_pred, "Raw (uncalibrated)")

print("\n=== Sample (30 before calibration) ===")
print(final_df.sample(min(30,len(final_df)), random_state=42)
      [["Participant","Focused","Predicted_Focused"]]
      .to_string(index=False))

# ---------------- Linear calibration (safe with NaNs) ----------------
valid = y_true.notna() & y_pred.notna()
if valid.sum() >= 3 and np.nanstd(y_pred[valid]) > 1e-6:
    reg = LinearRegression()
    reg.fit(y_pred[valid].values.reshape(-1,1), y_true[valid].values)
    y_cal = np.full(len(y_pred), np.nan, dtype=float)
    y_cal[valid] = reg.predict(y_pred[valid].values.reshape(-1,1))
    y_cal = np.clip(y_cal, 2.0, 7.0)  # keep in judges' range
    final_df["Calibrated_Focused"] = y_cal
    _, cal_stats = report(y_true, final_df["Calibrated_Focused"], "After Linear Calibration")
else:
    final_df["Calibrated_Focused"] = np.nan
    cal_stats = None
    print("\n[Note] Not enough valid rows/variance for calibration; skipped.")

print("\n=== Sample (30 after calibration) ===")
print(final_df.sample(min(30,len(final_df)), random_state=123)
      [["Participant","Focused","Predicted_Focused","Calibrated_Focused"]]
      .to_string(index=False))

# ---------------- Save ----------------
final_df.to_csv(OUT_PATH, index=False)
with open(METRICS_PATH, "w") as f:
    json.dump({
        "model": used_model,
        "raw": raw_stats,
        "calibrated": cal_stats
    }, f, indent=2)
print(f"\n💾 Saved results to: {OUT_PATH}")
print(f"📝 Saved metrics to: {METRICS_PATH}")
