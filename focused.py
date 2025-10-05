# ======================
# 1. Install dependencies (run this once manually in terminal)
# ======================
# pip install transformers accelerate scikit-learn openpyxl tqdm matplotlib

# ======================
# 2. Imports
# ======================
import pandas as pd
import torch
from transformers import pipeline
import re
from sklearn.metrics import r2_score
from tqdm import tqdm

# ======================
# 3. Load Data
# ======================
transcripts_path = "/home/labuser/research/VLM-Project/data/Transcripts_All.xlsx"
scores_path = "/home/labuser/research/VLM-Project/data/All_Scores.xlsx"

transcripts_df = pd.read_excel(transcripts_path)
scores_df = pd.read_excel(scores_path)

print("Transcript columns:", transcripts_df.columns)
print("Score columns:", scores_df.columns)

# ======================
# 4. Setup Hugging Face Model
# ======================
model_name = "google/flan-t5-large"
device = 0 if torch.cuda.is_available() else -1

generator = pipeline(
    "text2text-generation",
    model=model_name,
    device=device,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# ======================
# 5. Helper function: chunk transcript
# ======================
def chunk_transcript(transcript, max_words=80):
    """
    Split transcript into chunks of ~max_words each.
    Uses | separator to split by speaker turns.
    """
    parts = transcript.split("|")
    chunks = []
    current_chunk = ""
    current_count = 0

    for part in parts:
        words = part.split()
        if current_count + len(words) > max_words:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = part
            current_count = len(words)
        else:
            current_chunk += " " + part
            current_count += len(words)

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

# ======================
# 6. Helper function: predict score for a chunk (Focused)
# ======================
def predict_chunk_score(chunk):
    prompt = f"""
You are an expert communication evaluator.
Analyze the following interview transcript and score it from 0 to 9 for Focused.
Return ONLY a number (can be decimal).

Transcript:
{chunk}
"""
    try:
        output = generator(prompt, max_new_tokens=50, do_sample=False)[0]["generated_text"]
        # Extract first valid number <=9
        matches = re.findall(r"\d+(?:\.\d+)?", output)
        if matches:
            scores = [float(m) for m in matches if m.strip() != "" and float(m) <= 9]
            return scores[0] if scores else None
        return None
    except Exception as e:
        print("Error:", e)
        return None

# ======================
# 7. Run Inference for all participants
# ======================
results = []

for i, row in tqdm(transcripts_df.iterrows(), total=len(transcripts_df)):
    participant = row["Participant"]
    transcript = row["Transcripts"]

    # 1. Chunk transcript
    chunks = chunk_transcript(transcript, max_words=80)

    # 2. Predict each chunk
    chunk_scores = [predict_chunk_score(c) for c in chunks]

    # 3. Aggregate scores (average)
    valid_scores = [s for s in chunk_scores if s is not None]
    final_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

    results.append({
        "Participant": participant,
        "Predicted_Focused": final_score
    })

pred_df = pd.DataFrame(results)

# ======================
# 8. Merge with actual scores (Focused)
# ======================
final_df = pred_df.merge(
    scores_df[["Participant", "Focused"]],
    on="Participant",
    how="left"
)

# ======================
# 9. Show Predicted vs Actual for first 30 participants
# ======================
print(final_df[["Participant", "Predicted_Focused", "Focused"]].head(30))

# ======================
# 10. Calculate R² score (raw predictions vs actual Focused)
# ======================
r2 = r2_score(final_df["Focused"], final_df["Predicted_Focused"])
print("\nR² Score (raw predictions vs Focused):", r2)

# ======================
# 11. Linear Regression: Predicted vs Actual (Focused)
# ======================
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Drop rows with missing values
clean_df = final_df.dropna(subset=["Predicted_Focused", "Focused"])

X = clean_df[["Predicted_Focused"]].values
y = clean_df["Focused"].values

# Fit linear regression
reg = LinearRegression()
reg.fit(X, y)

# Predictions
y_pred = reg.predict(X)

# Compute R²
r2_linreg = r2_score(y, y_pred)

print("\nLinear Regression Results (Focused)")
print("===================================")
print(f"Coefficient (slope): {reg.coef_[0]:.4f}")
print(f"Intercept: {reg.intercept_:.4f}")
print(f"R² (linear regression fit): {r2_linreg:.4f}")

# ======================
# 12. Visualization
# ======================
plt.figure(figsize=(8,6))
plt.scatter(y, y_pred, alpha=0.7, label="Data points")
plt.plot([0, 9], [0, 9], 'r--', label='Perfect Prediction Line (y=x)')
plt.xlabel("Actual Focused Score")
plt.ylabel("Predicted Focused (Linear Fit)")
plt.title("Predicted vs Actual (Focused) — Linear Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
