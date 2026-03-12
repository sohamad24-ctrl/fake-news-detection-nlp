import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

# 1. Load cleaned dataset
df = pd.read_csv("../dataset/news_clean.csv")

# Features (text) and labels (0 = fake, 1 = real)
X = df["clean_text"].astype(str)
y = df["label"]

# 2. Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy(Before Calibration): {acc * 100:.2f}%")
print("\nClassification report:\n", classification_report(y_test, y_pred))

# 6. Save model and vectorizer
dump(vectorizer, "../models/tfidf_vectorizer.joblib")
dump(model, "../models/logreg_model.joblib")

print("\n✅ Saved model and vectorizer in the models/ folder.")
