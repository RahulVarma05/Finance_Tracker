import pandas as pd
import re
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Configuration
USE_CALIBRATION = True
CONFIDENCE_THRESHOLD = 0.80

# Fix 2: Add Random Seed Consistency
np.random.seed(42)
import random
random.seed(42)

# 1. Load Data
def load_data(filepath='transactions.csv'):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples.")
    if len(df) < 1000:
        print("\n⚠️  WARNING: Dataset size is small (< 1000 samples).")
        print("    Accuracy may be poor. Please regenerate data with: python generate_data.py --samples 3000")
        import time
        time.sleep(2) # Give user a moment to see warning
    return df

# 2. Preprocessing
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

# 3. Training & Evaluation
def train_and_evaluate():
    # ... (imports omitted) ...
    try:
        import train_model
        dataset_preprocessor = train_model.preprocess_data
    except ImportError:
        dataset_preprocessor = preprocess_data

    df = load_data()
    
    # Separation: Split Data ON RAW TEXT
    # Fix 4: Add Comment on Why Preprocessing Is in Pipeline
    # Why Pipeline?
    # 1. Prevents Leakage: Statistics (like IDF) are computed only on training data.
    # 2. Encapsulates Logic: Raw text goes in, prediction comes out. Safe for inference.
    # 3. Consistency: Ensures exactly the same regex cleanup is applied in Train and Inference.
    X = df['text']
    y = df['category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n--- Model Training ---")
    
    # Define Vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english')
    
    # Define Preprocessor Transformer
    preprocessor = FunctionTransformer(dataset_preprocessor, validate=False)

    # Model 1: Logistic Regression (Production Ready)
    print("\nTraining Logistic Regression...")
    
    base_clf = LogisticRegression(random_state=42, max_iter=1000)
    
    if USE_CALIBRATION:
        print("Enabling Probability Calibration (Sigmoid)...")
        # Wrap in CalibratedClassifierCV
        final_clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv=3)
    else:
        final_clf = base_clf
        
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor), 
        ('tfidf', vectorizer),
        ('clf', final_clf)
    ])
    lr_pipeline.fit(X_train, y_train)
    
    # Prediction (on Raw Text)
    y_pred_lr = lr_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_lr)
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")
    
    # ... (Metrics omitted) ...
    
    # Select Best Model for Production
    best_model = lr_pipeline
    
    # 4. Confidence Scoring Example
    print("\n--- Confidence Scoring Example ---")
    
    if hasattr(lr_pipeline.named_steps['clf'], "predict_proba"):
        probs = lr_pipeline.predict_proba(X_test)
        max_probs = np.max(probs, axis=1)
        
        # Validation checks with Threshold
        # Fix 3: Clarify Confidence Threshold
        low_confidence_indices = np.where(max_probs < CONFIDENCE_THRESHOLD)[0]
        
        print(f"Total Test Samples: {len(X_test)}")
        print(f"Samples with Confidence < {CONFIDENCE_THRESHOLD}: {len(low_confidence_indices)}")
        
        if len(low_confidence_indices) > 0:
            print("Example Low Confidence Predictions:")
            for idx in low_confidence_indices[:5]:
                # X_test is a series, use iloc
                text = X_test.iloc[idx]
                pred = y_pred_lr[idx]
                conf = max_probs[idx]
                print(f"Text: '{text}' -> Pred: {pred} (Conf: {conf:.2f})")
    else:
        print("Selected model does not support probability estimates. Confidence scoring skipped.")

    # 5. Save Model & Metadata (No Versioning)
    print("\n--- Saving Model & Metadata ---")
    
    import json
    from datetime import datetime

    latest_model_file = 'financial_model.pkl'

    # Save Latest Model
    joblib.dump(best_model, latest_model_file)
    print(f"Saved model: {latest_model_file}")

    # Create & Save Metadata
    # Convert params to string to avoid JSON serialization errors (e.g. numpy types)
    vectorizer_params = {k: str(v) for k, v in vectorizer.get_params().items()}
    
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "accuracy": float(f"{accuracy:.4f}"),
        "dataset_size": len(df),
        "calibration_enabled": USE_CALIBRATION,
        "train_test_split_ratio": 0.2,
        "random_seed": 42,
        "model_type": "LogisticRegression",
        "vectorizer_params": vectorizer_params
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    print("Saved metadata: model_metadata.json")

if __name__ == "__main__":
    train_and_evaluate()
