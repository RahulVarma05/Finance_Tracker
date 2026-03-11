import pandas as pd
import re
import numpy as np
import joblib
import sys
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

sys.stdout.reconfigure(encoding='utf-8')

# Configuration
USE_CALIBRATION      = True
CONFIDENCE_THRESHOLD = 0.80

np.random.seed(42)
import random
random.seed(42)

# ── 1. Load Data ──────────────────────────────────────────────────────────────
def load_data(filepath='transactions.csv'):
    df = pd.read_csv(filepath)
    if len(df) < 1000:
        # Prevent accidental under-training with insufficient data.
        raise ValueError(
            "Dataset too small (<1000 samples). "
            "Run: python generate_data.py --samples 3000 --imbalance"
        )
    # Filter out very short or non-alphanumeric garbage descriptions
    df = df[df['text'].astype(str).str.strip().str.len() >= 3]
    df = df[df['text'].astype(str).str.contains(r'[a-zA-Z0-9]', regex=True)]
    print(f"Loaded {len(df)} samples after cleaning.")
    return df

# ── 2. Preprocessing ──────────────────────────────────────────────────────────
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[₹$€£]|rs\.?|inr', '', text)
    text = re.sub(r'\d+(\.\d+)?', '<amount>', text)
    text = re.sub(r'[^a-z\s<>]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(X):
    """Wrapper for FunctionTransformer to process an iterable."""
    return [preprocess_text(x) for x in X]

# ── 3. Pipeline Builder ───────────────────────────────────────────────────────
def build_pipeline(clf, preprocessor, ngram=(1, 2), max_feat=15000, sublinear=True):
    vectorizer = TfidfVectorizer(
        ngram_range=ngram,
        max_features=max_feat,
        sublinear_tf=sublinear,   # log-scale term freq: improves accuracy on skewed text
        min_df=1,
        stop_words='english'
    )
    return Pipeline([
        ('preprocessor', preprocessor),
        ('tfidf',        vectorizer),
        ('clf',          clf)
    ])

# ── 4. Train & Evaluate ───────────────────────────────────────────────────────
def train_and_evaluate():
    dataset_preprocessor = preprocess_data

    df = load_data()
    X  = df['text']
    y  = df['category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    preprocessor = FunctionTransformer(dataset_preprocessor, validate=False)

    print("\n--- Model Training ---")

    # Model A: Logistic Regression (tuned – higher C, saga solver)
    print("\nTraining Logistic Regression (C=5, saga solver)...")
    lr_base = LogisticRegression(C=5.0, solver='saga', max_iter=2000, random_state=42)
    if USE_CALIBRATION:
        print("Enabling Probability Calibration (Sigmoid, cv=5)...")
        lr_clf = CalibratedClassifierCV(lr_base, method='sigmoid', cv=5)
    else:
        lr_clf = lr_base
    lr_pipeline = build_pipeline(lr_clf, preprocessor)
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)
    lr_acc  = accuracy_score(y_test, lr_pred)
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

    # Model B: Calibrated LinearSVC (strong linear baseline)
    print("\nTraining Calibrated LinearSVC (C=1.0)...")
    svm_base = LinearSVC(C=1.0, max_iter=3000, random_state=42)
    svm_clf  = CalibratedClassifierCV(svm_base, method='sigmoid', cv=5)
    svm_pipeline = build_pipeline(svm_clf, preprocessor)
    svm_pipeline.fit(X_train, y_train)
    svm_pred = svm_pipeline.predict(X_test)
    svm_acc  = accuracy_score(y_test, svm_pred)
    print(f"Calibrated LinearSVC Accuracy: {svm_acc:.4f}")

    # Pick best model
    if svm_acc >= lr_acc:
        best_model, best_pred, best_acc, best_name = svm_pipeline, svm_pred, svm_acc, "CalibratedLinearSVC"
    else:
        best_model, best_pred, best_acc, best_name = lr_pipeline, lr_pred, lr_acc, "LogisticRegression"

    print(f"\n>> Best Model: {best_name}  (Accuracy: {best_acc:.4f})")

    # Full per-category report
    print("\n--- Per-Category Report ---")
    print(classification_report(y_test, best_pred))

    # Confidence scoring
    print("--- Confidence Scoring ---")
    if hasattr(best_model.named_steps['clf'], "predict_proba"):
        probs     = best_model.predict_proba(X_test)
        max_probs = np.max(probs, axis=1)
        low_conf  = np.where(max_probs < CONFIDENCE_THRESHOLD)[0]
        print(f"Total Test Samples : {len(X_test)}")
        print(f"High Confidence    : {len(X_test) - len(low_conf)}")
        print(f"Low  Confidence    : {len(low_conf)}")
        if len(low_conf) > 0:
            print("\nExample Low Confidence Predictions:")
            for idx in low_conf[:5]:
                print(f"  '{X_test.iloc[idx]}' -> {best_pred[idx]} ({max_probs[idx]:.2f})")

    # Save
    print("\n--- Saving Model & Metadata ---")
    joblib.dump(best_model, 'financial_model.pkl')
    print("Saved model: financial_model.pkl")

    metadata = {
        "timestamp"             : datetime.now().isoformat(),
        "accuracy"              : float(f"{best_acc:.4f}"),
        "dataset_size"          : len(df),
        "model_type"            : best_name,
        "calibration_enabled"   : USE_CALIBRATION,
        "train_test_split_ratio": 0.2,
        "random_seed"           : 42,
        "tfidf_ngrams"          : "(1,2)",
        "tfidf_max_features"    : 15000,
        "tfidf_sublinear_tf"    : True
    }
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    print("Saved metadata: model_metadata.json")

if __name__ == "__main__":
    train_and_evaluate()
