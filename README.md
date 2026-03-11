# Financial Transaction Categorization & Extraction System

This project implements a comprehensive machine learning pipeline to categorize financial transaction text and extract precise transaction amounts, even from complex or ambiguous inputs (e.g., "Paid 500 wait no 600"). It supports both **text** and **voice** input.

## Architecture Overview

The system uses a **Dual-Model Architecture**:

1.  **Category Classifier (`financial_model.pkl`)**:
    *   **Algorithm**: Calibrated LinearSVC with TF-IDF Vectorization (n-grams 1–2).
    *   **Purpose**: Classifies transaction text into categories like *Food, Transport, Housing*.
    *   **Accuracy**: ~98% on clean data.

2.  **Amount Extractor (`amount_extractor.pkl`)**:
    *   **Algorithm**: Gradient Boosting Classifier.
    *   **Purpose**: Identifies the *correct* monetary amount from multiple candidates in the text.
    *   **Features**: Positional features, context windows, negation ("not 500"), correction ("sorry 600").
    *   **Accuracy**: ~99% transaction-level accuracy.

---

## Voice & STT Pipeline

The voice interface (`voice_inference.py`) follows this pipeline:

1.  **Capture**: Records audio via `sounddevice` (dual-threshold silence detection).
2.  **Transcribe**: Uses **OpenAI Whisper (Small Model)** for improved transcription accuracy.
3.  **Normalize**: Smart `word2number` logic converts phrases like "three hundred" to "300".
4.  **Correct**: Merchant name corrections (e.g., "swingy" → "swiggy").
5.  **Predict**: Passes cleaned text to the inference engine.

---

## Project Structure

| File | Purpose |
|---|---|
| `generate_data.py` | Generates synthetic transaction data |
| `import_real_data.py` | Imports & maps real-world bank CSV exports |
| `train_model.py` | Trains the Category Classifier |
| `train_amount_model.py` | Trains the Amount Extractor |
| `inference.py` | Core inference engine (text mode) |
| `voice_inference.py` | Voice wrapper with Whisper integration |
| `transactions.csv` | Training dataset |
| `dataset/` | Folder for real-world CSV files |

---

## How to Run Locally

### Step 1 — Install Dependencies
```bash
pip install pandas scikit-learn joblib openai-whisper sounddevice numpy word2number torch
```

### Step 2 — Generate Training Data
```bash
python generate_data.py --samples 5000 --imbalance
```
> Generates `transactions.csv` with 5000 labelled synthetic samples across 10 categories.

### Step 3 — (Optional) Add Real-World Data
Place your bank/UPI CSV exports in the `dataset/` folder, then run:
```bash
python import_real_data.py
```
> Merges real data into `transactions.csv` after cleaning and category mapping.

### Step 4 — Train Category Classifier
```bash
python train_model.py
```
> Trains both Logistic Regression & LinearSVC, picks best, saves `financial_model.pkl`.

### Step 5 — Train Amount Extractor
```bash
python train_amount_model.py
```
> Trains Gradient Boosting model, saves `amount_extractor.pkl`.

### Step 6 — Run the Model

**Text Mode** (type your transactions):
```bash
python inference.py
```

**Voice Mode** (speak your transactions):
```bash
python voice_inference.py
```

---

## Quick Reference
```bash
pip install pandas scikit-learn joblib openai-whisper sounddevice numpy word2number torch
python generate_data.py --samples 5000 --imbalance
python import_real_data.py        # optional
python train_model.py
python train_amount_model.py
python inference.py               # text mode
python voice_inference.py         # voice mode
```

---

## Categories Supported

| Category | Examples |
|---|---|
| Food | Swiggy, Zomato, restaurant, grocery |
| Transport | Uber, Ola, fuel, flight, metro |
| Housing | Rent, electricity, plumber, maintenance |
| Entertainment | Netflix, movies, concerts, games |
| Shopping | Amazon, Flipkart, clothes, electronics |
| Utilities | Jio, Airtel, broadband, DTH |
| Health | Doctor, medicine, gym, pharmacy |
| Education | Courses, books, tuition, workshops |
| Income | Salary, bonus, cashback, refunds |
| Others | Donations, ATM, transfers |

---

## Technical Details

### Confidence Thresholds
*   **>= 0.80** → High Confidence (result shown normally)
*   **0.50–0.79** → Medium Confidence (shown with warning)
*   **< 0.30** → Low Confidence (voice mode skips result, asks to repeat)

### Amount Extraction Logic
1.  **Strong Correction Override**: "sorry", "actually", "correction" → picks last corrected number
2.  **Negation Penalty**: "not", "instead of" → 90% probability reduction  
3.  **No-Cue Fallback**: If no semantic cues, picks **last** number (most likely intended)
4.  **Threshold Fallback**: If model confidence < 0.20, falls back to last number

---

## Improvement Plan

- **Retraining with User Corrections**: Store corrections to improve edge cases
- **Deep Learning**: Move to DistilBERT for more complex semantic understanding
- **FastAPI Deployment**: Wrap `predict_transaction` for mobile/web usage
- **Docker**: Containerize for cloud deployment
