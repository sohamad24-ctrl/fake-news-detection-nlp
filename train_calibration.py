import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import dump

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load your small, manually labelled dataset
df = pd.read_csv("../dataset/short_inputs.csv")

df["text_clean"] = df["text"].apply(clean_text)

X = df["text_clean"].values
y = df["label"].values   # 0 = fake, 1 = real

# Simple TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=500, class_weight="balanced")
model.fit(X_vec, y)

# Save calibration model + vectorizer
dump(vectorizer, "../models/calib_vectorizer.joblib")
dump(model, "../models/calib_model.joblib")

print("✅ Calibration model saved as calib_vectorizer.joblib and calib_model.joblib")
