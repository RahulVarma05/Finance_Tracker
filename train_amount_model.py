import pandas as pd
import numpy as np
import re
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# Fix 4: Add Seed for Reproducibility
np.random.seed(42)

# 1. Feature Extraction Logic
def extract_candidates_with_features(text):
    """
    Parses text, finds all numbers, and extracts features for each.
    Returns: List of dicts (one per candidate number)
    """
    # Regex for numbers (integers or floats, ignore commas)
    # Capture group for number to get position
    # We iterate manually to get spans
    matches = []
    for m in re.finditer(r'[\d,]+(?:\.\d+)?', text):
        val_str = m.group()
        # Clean commas
        clean_str = val_str.replace(',', '')
        try:
            val = float(clean_str)
            matches.append({
                "val": val,
                "start": m.start(),
                "end": m.end(),
                "text_str": val_str
            })
        except ValueError:
            continue
            
    if not matches:
        return []

    processed_candidates = []
    total_len = len(text)
    num_candidates = len(matches)
    
    # Correction Keywords
    # "Paid 300 wrong no 400" -> 400 preceded by "no", "wrong"
    correction_cues = ["sorry", "no", "wait", "actually", "correction", "read as"]
    # Negation Keywords
    # "Bill is 3000 not 3200", "Paid 500 instead of 600"
    negation_cues = ["not", "isn't", "is not", "instead of"]
    
    for i, m in enumerate(matches):
        features = {}
        features["val"] = m["val"]
        
        # Positional Features
        features["position_ratio"] = m["start"] / total_len if total_len > 0 else 0
        features["is_last"] = 1 if i == num_candidates - 1 else 0
        features["is_first"] = 1 if i == 0 else 0
        features["num_candidates"] = num_candidates
        
        # Context Window (Look behind 20 chars)
        start_lookback = max(0, m["start"] - 25)
        prebox = text[start_lookback:m["start"]].lower()
        
        # Preceded by correction? Using regex for word boundaries
        correction_pattern = r'\b(' + '|'.join([re.escape(w) for w in correction_cues]) + r')\b'
        features["preceded_by_correction"] = 1 if re.search(correction_pattern, prebox) else 0

        # Preceded by negation? (High chance of being WRONG)
        # Why Negation Logic?
        # "not", "isn't" are high-precision negative signals. 
        # If a number is negated, it is almost certainly NOT the target.
        negation_pattern = r'\b(' + '|'.join([re.escape(w) for w in negation_cues]) + r')\b'
        features["preceded_by_negation"] = 1 if re.search(negation_pattern, prebox) else 0
        
        # Preceded by currency?
        currency_pattern = r'(' + '|'.join([re.escape(w) for w in ["rs", "inr", "₹", "$"]]) + r')'
        features["preceded_by_currency"] = 1 if re.search(currency_pattern, prebox) else 0
        
        # Look ahead (Look ahead 20 chars) - mostly for "instead of 500" where 500 is wrong
        end_lookahead = min(len(text), m["end"] + 25)
        postbox = text[m["end"]:end_lookahead].lower()
        
        # Followed by correction?
        features["followed_by_correction"] = 1 if re.search(correction_pattern, postbox) else 0
        
        processed_candidates.append(features)
        
    return processed_candidates

# 2. Process Transactions to Feature Rows
def process_transactions(df):
    """
    Converts a dataframe of transactions into candidate feature rows.
    """
    X_rows = []
    y_rows = []
    
    for _, row in df.iterrows():
        text = str(row['text'])
        try:
            target = float(row['amount'])
        except ValueError:
            continue
            
        candidates = extract_candidates_with_features(text)
        if not candidates:
            continue
            
        # Check if any candidate matches target (fuzzy match)
        has_match = False
        for c in candidates:
            if abs(c["val"] - target) < 0.01:
                has_match = True
                break
        
        # Only use samples where the correct answer IS in the text
        if not has_match:
            continue
            
        # Add rows for this transaction
        for c in candidates:
            # Feature Vector
            feats = [
                c["position_ratio"],
                c["is_last"],
                c["is_first"],
                c["num_candidates"],
                c["preceded_by_correction"],
                c["preceded_by_negation"],
                c["preceded_by_currency"],
                c["followed_by_correction"]
            ]
            
            # Label
            label = 1 if abs(c["val"] - target) < 0.01 else 0
            
            X_rows.append(feats)
            y_rows.append(label)
            
    return np.array(X_rows), np.array(y_rows)

def train_amount_extractor():
    print("Loading dataset...")
    df = pd.read_csv('transactions.csv')
    print(f"Loaded {len(df)} samples.")
    if len(df) < 1000:
        # Prevent accidental under-training with insufficient data.
        raise ValueError(
            "Dataset too small (<1000). Regenerate using: "
            "python generate_data.py --samples 3000 --imbalance"
        )
    
    
    # --- CRITICAL FIX: TRANSATION-LEVEL SPLIT ---
    # We split the raw transactions first using train_test_split.
    # This prevents "Data Leakage" where candidates from the same transaction 
    # could otherwise appear in both Train and Test sets if we split after expansion.
    print("Splitting data at Transaction Level (80/20)...")
    
    # Use stratification to ensure all categories are represented in training
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['category'])
    
    print(f"Training Transactions: {len(train_df)}")
    print(f"Testing Transactions: {len(test_df)}")
    n_train_tx = len(train_df)
    n_test_tx = len(test_df)
    
    # Now expand into candidate rows
    print("Generating candidate features...")
    X_train, y_train = process_transactions(train_df)
    X_test, y_test = process_transactions(test_df)
    
    print(f"Train Candidate Rows: {len(X_train)}")
    print(f"Test Candidate Rows: {len(X_test)}")
    count_train_candidates = len(X_train)
    count_test_candidates = len(X_test)
    
    print("\nTraining Gradient Boosting Classifier...")
    # Why Gradient Boosting?
    # 1. Non-linear Interactions: Can learn complex relationships between position (is_last) and context context (preceded_by_correction).
    # 2. Ranking Capability: Effectively learns to rank candidates by probability score.
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    
    print("\nEvaluating...")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Feature Importance
    feature_names = ["position_ratio", "is_last", "is_first", "num_candidates", "preceded_by_correction", "preceded_by_negation", "preceded_by_currency", "followed_by_correction"]
    print("\nFeature Importances:")
    for name, imp in zip(feature_names, clf.feature_importances_):
        print(f"{name:<25}: {imp:.4f}")

    # --- Transaction-Level Accuracy ---
    print("\nCalculating Transaction-Level Accuracy...")
    correct_transactions = 0
    total_eval_transactions = 0
    
    for _, row in test_df.iterrows():
        text = str(row['text'])
        try:
            target = float(row['amount'])
        except ValueError:
            continue
            
        candidates = extract_candidates_with_features(text)
        if not candidates:
            continue
            
        # Filter: only evaluate if target is actually in candidates
        # (Otherwise it's an extraction failure, not a ranking failure, but effectively a fail)
        target_in_candidates = False
        for c in candidates:
            if abs(c["val"] - target) < 0.01:
                target_in_candidates = True
                break
        
        if not target_in_candidates:
            # If the correct amount isn't even found by regex, it's wrong.
            total_eval_transactions += 1
            continue
            
        # Predict
        X_feats = []
        for c in candidates:
            X_feats.append([
                c["position_ratio"],
                c["is_last"],
                c["is_first"],
                c["num_candidates"],
                c["preceded_by_correction"],
                c["preceded_by_negation"],
                c["preceded_by_currency"],
                c["followed_by_correction"]
            ])
            
        probas = clf.predict_proba(X_feats)[:, 1]
        
        # Apply Inference Heuristics (matches inference.py)
        for i, c in enumerate(candidates):
            if c["preceded_by_negation"]:
                probas[i] *= 0.1 # Soft Penalty
            if c["preceded_by_correction"]:
                probas[i] *= 1.5 # Stronger Boost (improved reliability)
            
        best_idx = np.argmax(probas)
        predicted_val = candidates[best_idx]["val"]
        
        if abs(predicted_val - target) < 0.01:
            correct_transactions += 1
            
        total_eval_transactions += 1
        
    acc = correct_transactions / total_eval_transactions if total_eval_transactions > 0 else 0
    print(f"Transaction-Level Accuracy: {acc:.4f} ({correct_transactions}/{total_eval_transactions})")
    
    # Why this metric?
    # Candidate-level accuracy (how many candidates are classified correctly as 0 or 1) is dominated by 0s (imbalanced).
    # Transaction-level accuracy measures the real-world utility: did we pick the SINGLE correct amount for the transaction?
        
    print("\nSaving Amount Model...")
    joblib.dump(clf, 'amount_extractor.pkl')
    print("Saved to 'amount_extractor.pkl'")

    # Save Metadata
    metadata = {
        "model_type": "GradientBoostingClassifier",
        "timestamp": datetime.now().isoformat(),
        "random_seed": 42,
        "train_test_split": "80/20",
        "train_transactions": n_train_tx,
        "test_transactions": n_test_tx,
        "candidate_accuracy": float(f"{accuracy_score(y_test, y_pred):.4f}"),
        "transaction_accuracy": float(f"{acc:.4f}"),
        "features": feature_names
    }
    
    with open("amount_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print("Saved metadata to 'amount_model_metadata.json'")

if __name__ == "__main__":
    train_amount_extractor()
