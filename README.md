# Financial Transaction Categorization & Extraction System

This project implements a comprehensive machine learning pipeline to categorize financial transaction text and extract precise transaction amounts, even from complex or ambiguous inputs (e.g., "Paid 500 wait no 600"). It supports both text and voice input.

## 🏗️ Architecture Overview

The system uses a **Dual-Model Architecture**:

1.  **Category Classifier (`financial_model.pkl`)**:
    *   **Algorithm**: Logistic Regression with TF-IDF Vectorization.
    *   **Purpose**: Classifies transaction text into categories like *Food, Transport, Housing*.
    *   **Features**: Calibrated probabilities for confidence scoring.

2.  **Amount Extractor (`amount_extractor.pkl`)**:
    *   **Algorithm**: Gradient Boosting Classifier.
    *   **Purpose**: Identifies the *correct* monetary amount from multiple candidates in the text.
    *   **Features**: Uses positional features, context windows, and linguistic cues (negation "not 500", correction "sorry 600").

## 🚀 Voice & STT Pipeline

The voice interface (`voice_inference.py`) follows this pipeline:

1.  **Capture**: Records audio via `sounddevice` (silence detection enabled).
2.  **Transcribe**: Uses **OpenAI Whisper (Base Model)** to convert speech to text.
3.  **Normalize**: Smart `word2number` logic converts phrases like "three hundred" to "300" without corrupting non-numeric text.
4.  **Influence**: Passes normalized text to the inference engine.

## 📂 Project Structure

- `generate_data.py`: Generates synthetic, complex transaction data (supports ambiguity/noise).
- `train_model.py`: Trains the Category Classifier.
- `train_amount_model.py`: Trains the Amount Extractor.
- `inference.py`: Core inference logic (loads models, applies heuristics).
- `voice_inference.py`: Voice wrapper with Whisper integration.
- `transactions.csv`: Generated dataset.

---

## ⚡ Instructions

### 1. Setup Environment
Install dependencies:
```bash
pip install pandas scikit-learn joblib openai-whisper sounddevice scipy numpy wavio word2number
```

### 2. Generate Data
**⚠️ Warning:** Training on small datasets (< 1000 samples) will yield poor results.
We recommend 3000 samples with class imbalance simulation:
```bash
python generate_data.py --samples 3000 --imbalance
```

### 3. Train Models
Train both models in this specific order:
```bash
python train_model.py
python train_amount_model.py
```

### 4. Run Inference

**Option A: Text Mode**
Interactive CLI for typing transactions:
```bash
python inference.py
```

**Option B: Voice Mode**
Speak your transaction (GPU auto-detected):
```bash
python voice_inference.py
```

---

## 🧠 Technical Details

### Transaction-Level Accuracy
For amount extraction, we optimize for **Transaction-Level Accuracy** rather than candidate-level accuracy. This metric measures how often the model correctly identifies the *single* true amount for a transaction, which is the only metric that matters for end-user utility.

### Confidence Thresholds & Heuristics
The system employs defensive heuristics for reliability:
*   **Negation Penalty**: Candidates preceded by "not", "instead of" have their probability reduced by **90%**.
*   **Confidence Threshold**: If the amount extractor's top confidence is below **0.20**, the system falls back to the **last detected number** (a common linguistic pattern for corrections).
*   **Safety Guards**: Returns `None` gracefully if no numbers are found, preventing crashes.

## 🔮 Improvement Plan (Future Work)

### 1. Retraining with User Feedback
- Store user corrections to improve handling of edge cases.
- Apply higher weights to corrected samples during retraining.

### 2. Advanced Handling
- **Class Imbalance**: Implement `class_weight='balanced'` or SMOTE for rare categories.
- **Deep Learning**: Move to DistilBERT if semantic complexity increases beyond TF-IDF capabilities.

### 3. Deployment
- Wrap `predict_transaction` in a FastAPI endpoint for real-time mobile/web usage.
- Dockerize the application for cloud deployment.
