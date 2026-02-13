import joblib
import numpy as np
import sys
import re
import train_model # Import module to register functions for pickle

# Load Model
try:
    model_pipeline = joblib.load('financial_model.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file 'financial_model.pkl' not found. Please run train_model.py first.")
    sys.exit(1)
except ImportError as e:
    print(f"Error loading model: {e}")
    print("Make sure train_model.py is in the same directory.")
    sys.exit(1)

# Load Amount Extractor Model
try:
    amount_model = joblib.load('amount_extractor.pkl')
    print("Amount Extractor loaded successfully.")
except FileNotFoundError:
    amount_model = None
    print("Amount Extractor not found. Using fallback logic.")

def extract_candidates_with_features(text):
    """
    Parses text, finds all numbers, and extracts features for each.
    Returns: List of dicts (one per candidate number)
    """
    matches = []
    for m in re.finditer(r'[\d,]+(?:\.\d+)?', text):
        val_str = m.group()
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
    
    correction_cues = ["sorry", "no", "wait", "actually", "correction", "read as"]
    strong_correction_cues = ["sorry", "actually", "correction", "read as"]
    negation_cues = ["not", "isn't", "is not", "instead of"]
    
    for i, m in enumerate(matches):
        features = {}
        features["val"] = m["val"]
        features["position_ratio"] = m["start"] / total_len if total_len > 0 else 0
        features["is_last"] = 1 if i == num_candidates - 1 else 0
        features["is_first"] = 1 if i == 0 else 0
        features["num_candidates"] = num_candidates
        
        start_lookback = max(0, m["start"] - 25)
        prebox = text[start_lookback:m["start"]].lower()
        
        # Preceded by correction? Using regex for word boundaries
        correction_pattern = r'\b(' + '|'.join([re.escape(w) for w in correction_cues]) + r')\b'
        features["preceded_by_correction"] = 1 if re.search(correction_pattern, prebox) else 0
        
        # PROMPT: Strong correction detection
        strong_pattern = r'\b(' + '|'.join([re.escape(w) for w in strong_correction_cues]) + r')\b'
        features["is_strong_correction"] = 1 if re.search(strong_pattern, prebox) else 0
        
        # Preceded by negation?
        negation_pattern = r'\b(' + '|'.join([re.escape(w) for w in negation_cues]) + r')\b'
        features["preceded_by_negation"] = 1 if re.search(negation_pattern, prebox) else 0
        
        # Preceded by currency?
        currency_pattern = r'(' + '|'.join([re.escape(w) for w in ["rs", "inr", "₹", "$"]]) + r')'
        features["preceded_by_currency"] = 1 if re.search(currency_pattern, prebox) else 0
        
        end_lookahead = min(len(text), m["end"] + 25)
        postbox = text[m["end"]:end_lookahead].lower()
        
        features["followed_by_correction"] = 1 if re.search(correction_pattern, postbox) else 0
        
        processed_candidates.append(features)
        
    return processed_candidates

def extract_amount(text):
    """
    Extracts the potential transaction amount from text using ML model.
    """
    # 1. Feature Extraction
    candidates = extract_candidates_with_features(text)
    if not candidates:
        return None
        
    # 2. Model Prediction
    if amount_model:
        # Rule-based override: Detect strong correction cues
        # Rule-based override improves reliability for complex corrections.
        strong_candidates = [c for c in candidates if c.get("is_strong_correction")]
        if strong_candidates:
             return strong_candidates[-1]["val"]

        # PROMPT: No-cue fallback (Last Number Rule)
        # If multiple numbers exist but NO cues (negation/correction), assume simple sequence -> last is valid.
        has_cues = any(c.get("preceded_by_correction") or c.get("preceded_by_negation") for c in candidates)
        if not has_cues and len(candidates) > 1:
             # When no semantic cues exist, last number is most likely intended amount.
             return candidates[-1]["val"]

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
            
        # Predict Proba
        try:
            # We want class 1 (is_target) probability
            probas = amount_model.predict_proba(X_feats)[:, 1]
            
            # Rule-based Override: Weighted Logic
            # Penalize negation/boost correction
            for i, c in enumerate(candidates):
                if c["preceded_by_negation"]:
                    probas[i] *= 0.1 # Soft Penalty (Matches Training)
                if c["preceded_by_correction"]:
                    probas[i] *= 1.5 # Stronger Boost (improved reliability)
                    
            # Ranking Strategy: Model Probability + Heuristics
            # We combine the model's confidence with rule-based multipliers (negation, correction).
            best_idx = np.argmax(probas)
            
            # Confidence Threshold Logic
            # If the model is unsure about ANY candidate (prob < 0.20), we fallback to the last number.
            # This is a heuristic: often the last number is the final/corrected amount.
            AMOUNT_PROB_THRESHOLD = 0.20
            
            if np.max(probas) < AMOUNT_PROB_THRESHOLD:
                 return candidates[-1]["val"]
                 
            return candidates[best_idx]["val"]
        except Exception as e:
            print(f"Error in amount prediction: {e}")
            pass # Fallback

    # 3. Fallback (Last Number)
    # If model fails or not loaded, return the last number found (most common pattern in simple texts).
    return candidates[-1]["val"] if candidates else None

def predict_transaction(text):
    """
    Predicts category and extracts amount.
    Returns: (category: str, confidence: float, amount: float | None)
    """
    # Pass raw text list to pipeline
    input_data = [text] 
    
    # Predict
    prediction = model_pipeline.predict(input_data)[0]
    
    # Safe guard
    # Check if the underlying classifier supports predict_proba
    if hasattr(model_pipeline.named_steps["clf"], "predict_proba"):
        probabilities = model_pipeline.predict_proba(input_data)[0]
        confidence = np.max(probabilities)
    else:
        # Fallback for models like SVM/LinearSVC without probability calibration
        try:
            # Some models have decision_function
            decision = model_pipeline.decision_function(input_data)[0]
            # Simple normalization for confidence proxy (not a real probability)
            confidence = 1.0 / (1.0 + np.exp(-np.max(np.abs(decision)))) 
        except:
            confidence = 0.0
    
    # Extract Amount
    amount = extract_amount(text)
    
    return prediction, confidence, amount

if __name__ == "__main__":
    print("\n--- Financial Transaction Classifier ---")
    print("Type a transaction description to classify (or 'exit' to quit).")
    
    while True:
        try:
            user_input = input("\nEnter transaction: ")
            if user_input.lower() in ('exit', 'quit'):
                break
            if not user_input.strip():
                continue
                
            cat, conf, amount = predict_transaction(user_input)
            # Determine Status
            if conf >= 0.80:
                status = "High Confidence"
            elif conf >= 0.50:
                status = "Medium Confidence"
            else:
                status = "Low Confidence"

            amount_str = f"₹{amount}" if amount is not None else "Not found"
            
            print("-" * 30)
            print(f"Category:   {cat}")
            print(f"Confidence: {conf:.2f}")
            print(f"Status:     {status}")
            print(f"Amount:     {amount_str}")
            print("-" * 30)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
