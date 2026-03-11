"""
import_real_data.py
-------------------
Imports real-world transaction datasets from the /dataset folder,
maps their categories to our 10 model categories, combines them
with any existing synthetic data, deduplicates, and saves to
transactions.csv ready for training.
"""

import pandas as pd
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ── Category Mapping ─────────────────────────────────────────────────────────
# Maps raw category strings (from real datasets) → our 10 model categories.
CATEGORY_MAP = {
    # Food
    "food & drink"       : "Food",
    "groceries"          : "Food",
    "restaurants"        : "Food",
    "coffee shops"       : "Food",
    "fast food"          : "Food",
    "alcohol & bars"     : "Food",
    "food & dining"      : "Food",

    # Transport
    "travel"             : "Transport",
    "gas & fuel"         : "Transport",
    "auto insurance"     : "Transport",

    # Housing
    "rent"               : "Housing",
    "mortgage & rent"    : "Housing",
    "home improvement"   : "Housing",

    # Entertainment
    "entertainment"      : "Entertainment",
    "music"              : "Entertainment",
    "movies & dvds"      : "Entertainment",
    "television"         : "Entertainment",

    # Shopping
    "shopping"           : "Shopping",
    "electronics & software": "Shopping",

    # Utilities
    "utilities"          : "Utilities",
    "mobile phone"       : "Utilities",
    "internet"           : "Utilities",

    # Health
    "health & fitness"   : "Health",
    "haircut"            : "Health",

    # Education  (none in these datasets, placeholder for future)
    "education"          : "Education",

    # Income
    "salary"             : "Income",
    "paycheck"           : "Income",

    # Investment
    "investment"         : "Investment",

    # Others / skip
    "other"              : "Others",
    "credit card payment": None,   # Filtered out – not a real expense category
}

FINANCIAL_KEYWORDS = {
    # Merchants & Apps
    "amazon", "flipkart", "swiggy", "zomato", "uber", "ola", "netflix", "spotify",
    "airtel", "jio", "hdfc", "sbi", "icici", "paytm", "phonepe", "gpay",
    "myntra", "ajio", "irctc", "makemytrip", "bigbasket", "blinkit",
    "zerodha", "groww", "upstox", "binance", "wazirx",
    # Transaction words
    "paid", "payment", "bill", "rent", "salary", "grocery", "groceries",
    "subscription", "recharge", "transfer", "purchase", "order", "bought",
    "spent", "fee", "charge", "debit", "credit", "invoice", "receipt",
    "restaurant", "food", "dining", "cafe", "shop", "store", "medical",
    "doctor", "medicine", "gym", "fitness", "fuel", "petrol", "electricity",
    "internet", "mobile", "insurance", "emi", "loan", "mortgage", "refund",
    "cashback", "bonus", "travel", "flight", "hotel", "ticket",
    "invest", "invested", "investment", "stocks", "shares", "equity", "mutual",
    "sip", "crypto", "bitcoin", "bonds", "deposit", "fd"
}

def is_financial_text(text):
    """Return True if description contains at least one financial keyword."""
    if not isinstance(text, str):
        return False
    words = set(text.lower().split())
    return bool(words & FINANCIAL_KEYWORDS)

# ── Loader: Personal_Finance_Dataset.csv ─────────────────────────────────────
def load_finance_dataset(path):
    df = pd.read_csv(path)
    # Columns: Date, Transaction Description, Category, Amount, Type
    df = df.rename(columns={
        "Transaction Description": "text",
        "Category": "raw_category",
        "Amount": "amount"
    })
    # Keep only Expense rows
    if "Type" in df.columns:
        df = df[df["Type"].str.lower() == "expense"]

    # Filter out random/gibberish descriptions (random text hurts model accuracy)
    before = len(df)
    df = df[df["text"].apply(is_financial_text)]
    print(f"   Filtered {before - len(df)} gibberish rows from Personal_Finance_Dataset.")

    return df[["text", "raw_category", "amount"]]

# ── Loader: personal_transactions.csv ────────────────────────────────────────
def load_personal_transactions(path):
    df = pd.read_csv(path)
    # Columns: Date, Description, Amount, Transaction Type, Category, Account Name
    df = df.rename(columns={
        "Description": "text",
        "Category":    "raw_category",
        "Amount":      "amount"
    })
    # Keep only debit (expense) rows
    if "Transaction Type" in df.columns:
        df = df[df["Transaction Type"].str.lower() == "debit"]
    return df[["text", "raw_category", "amount"]]

# ── Map + Clean ───────────────────────────────────────────────────────────────
def map_categories(df):
    df["raw_lower"] = df["raw_category"].str.lower().str.strip()
    df["category"]  = df["raw_lower"].map(CATEGORY_MAP)

    # Report unmapped
    unmapped = df[df["category"].isna()]["raw_category"].unique()
    if len(unmapped):
        print(f"⚠️  Unmapped categories (will be dropped): {list(unmapped)}")

    # Drop rows with no mapping (None or NaN)
    df = df.dropna(subset=["category"])
    df = df[df["category"].notna()]

    return df[["text", "category", "amount"]]

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    DATASET_DIR  = "dataset"
    OUTPUT_CSV   = "transactions.csv"

    all_frames = []

    # 1. Load real datasets
    f1 = os.path.join(DATASET_DIR, "Personal_Finance_Dataset.csv")
    f2 = os.path.join(DATASET_DIR, "personal_transactions.csv")

    for path, loader in [(f1, load_finance_dataset), (f2, load_personal_transactions)]:
        if os.path.exists(path):
            print(f"📂 Loading: {path}")
            df = loader(path)
            df = map_categories(df)
            print(f"   → {len(df)} usable rows after mapping")
            all_frames.append(df)
        else:
            print(f"⚠️  File not found, skipping: {path}")

    if not all_frames:
        print("❌ No data loaded. Exiting.")
        sys.exit(1)

    real_df = pd.concat(all_frames, ignore_index=True)

    # 2. Merge with existing synthetic data (if any)
    if os.path.exists(OUTPUT_CSV):
        print(f"\n📂 Merging with existing {OUTPUT_CSV}...")
        existing = pd.read_csv(OUTPUT_CSV)
        # Ensure existing has same columns
        if "text" in existing.columns and "category" in existing.columns:
            existing = existing[["text", "category", "amount"]] if "amount" in existing.columns else existing[["text", "category"]].assign(amount=None)
            combined = pd.concat([existing, real_df], ignore_index=True)
        else:
            print("⚠️  Existing transactions.csv has unexpected format – using real data only.")
            combined = real_df
    else:
        combined = real_df

    # 3. Drop duplicates
    before = len(combined)
    combined.drop_duplicates(subset=["text", "category"], inplace=True)
    after = len(combined)
    print(f"\n🗑️  Removed {before - after} duplicates.")

    # 4. Validate dataset size
    if len(combined) < 1000:
        print(f"⚠️  Warning: Only {len(combined)} samples. Consider adding more data.")
    
    # 5. Save
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved {len(combined)} samples to {OUTPUT_CSV}")

    # 6. Category distribution
    print("\n📊 Final Category Distribution:")
    print(combined["category"].value_counts().to_string())

    print("\n🚀 Now run training:")
    print("   python train_model.py")
    print("   python train_amount_model.py")

if __name__ == "__main__":
    main()
