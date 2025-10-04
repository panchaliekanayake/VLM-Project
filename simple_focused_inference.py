# ======= simple_one_participant_inference.py =======
import os
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer

# ---- file paths (correct) ----
TRANSCRIPTS_PATH = "/home/labuser/research/VLM-Project/data/Transcripts_All.xlsx"

def main():
    # ---- check file ----
    if not os.path.exists(TRANSCRIPTS_PATH):
        print(f"Data file not found at {TRANSCRIPTS_PATH}")
        return

    # ---- load data ----
    df = pd.read_excel(TRANSCRIPTS_PATH)
    if df.empty:
        print("Transcripts file is empty")
        return

    # ---- normalize columns & handle trailing-space header ----
    df.columns = [c.strip() for c in df.columns]
    if "Participant" not in df.columns:
        raise KeyError(f"'Participant' column not found. Available: {list(df.columns)}")
    # accept either 'Transcripts' or 'Transcripts ' (trailing space)
    transcript_col = "Transcripts" if "Transcripts" in df.columns else None
    if transcript_col is None and "Transcripts " in df.columns:
        df.rename(columns={"Transcripts ": "Transcripts"}, inplace=True)
        transcript_col = "Transcripts"
    if transcript_col is None:
        # last resort: try a few common names
        for alt in ["Transcript", "Text", "Interview", "Transcript_Text"]:
            if alt in df.columns:
                df.rename(columns={alt: "Transcripts"}, inplace=True)
                transcript_col = "Transcripts"
                break
    if transcript_col is None:
        raise KeyError(f"No transcript column found. Available: {list(df.columns)}")

    # ---- pick first non-empty transcript ----
    row = df[df["Transcripts"].astype(str).str.strip().ne("")].iloc[0]
    participant = row["Participant"]
    transcript  = str(row["Transcripts"])

    # ---- load model (small, fast) ----
    device = 0 if torch.cuda.is_available() else -1
    model_name = "microsoft/phi-3-mini-4k-instruct"  # good default
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        pad_id = getattr(tok, "pad_token_id", getattr(tok, "eos_token_id", None))
        generator = pipeline(
            "text-generation",
            model=model_name,
            device=device,
            torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
            max_new_tokens=32,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=pad_id,
            return_full_text=False
        )
    except Exception as e:
        print(f"Primary model failed ({model_name}): {e}")
        # fallback that downloads smaller weights
        model_name = "google/flan-t5-small"
        generator = pipeline("text2text-generation", model=model_name, device=device)

    print(f"✅ Model used: {model_name}")

    # ---- prompt (Focused, 0–9) ----
    prompt = (
        "You are an evaluator.\n"
        "Give ONE overall score for the Interviewee’s **Focused** quality (0–9, decimals allowed).\n"
        "Focused = stays on-topic, directly answers; clear, organized, logical; relevant details; avoids tangents.\n"
        "Ignore pleasantries/fillers. Return only a number between 0 and 9.\n\n"
        f"Transcript:\n{transcript}\n\nScore:"
    )

    # ---- generate ----
    print("Generating...")
    out = generator(prompt)
    text = out[0]["generated_text"] if isinstance(out, list) and "generated_text" in out[0] else str(out)

    # ---- save ----
    out_file = "/home/labuser/research/VLM-Project/data/simple_inference_result.csv"
    pd.DataFrame([{"Participant": participant, "Model": model_name, "LLM_Output": text}]).to_csv(out_file, index=False)
    print(f"💾 Saved result to: {out_file}")
    print("Preview:", text)

if __name__ == "__main__":
    main()
