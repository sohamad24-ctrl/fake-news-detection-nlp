import re
from joblib import load

# Load ONLY the calibration model (small, clean one)
calib_vectorizer = load("../models/calib_vectorizer.joblib")
calib_model = load("../models/calib_model.joblib")

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_news(text: str) -> str:
    cleaned = clean_text(text)
    X_vec = calib_vectorizer.transform([cleaned])
    proba = calib_model.predict_proba(X_vec)[0]  # [P(fake), P(real)]

    p_fake = proba[0]
    p_real = proba[1]

    # Threshold-based decision
    if p_fake >= 0.55:
        label = "FAKE NEWS ❌"
    elif p_real >= 0.55:
        label = "REAL NEWS ✅"
    else:
        label = "UNCERTAIN 🤔"

    return f"{label}  (P(fake)={p_fake:.2f}, P(real)={p_real:.2f})"




if __name__ == "__main__":
    print("Calibrated Fake News Demo (short sentences)")
    print("Type a news-like sentence. Type 'exit' to quit.\n")

    while True:
        text = input("Enter news text: ")
        if text.lower().strip() in ("exit", "quit"):
            print("Bye!")
            break
        print("Result:", predict_news(text), "\n")
