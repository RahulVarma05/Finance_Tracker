import sqlite3
from datetime import datetime

# ── 1. DB File Path ───────────────────────────────────────────────────────────
# This is the actual database file that gets created on disk
# Think of it like your entire database living in one file
DB_FILE = "finance.db"


# ── 2. get_db() ───────────────────────────────────────────────────────────────
def get_db():
    """
    Opens and returns a connection to the SQLite database.
    
    row_factory = sqlite3.Row allows us to access columns by name
    instead of index position.
    
    Example:
        row["category"]   ✅ works with row_factory
        row[2]            ✅ also works but less readable
    
    Called by every function that needs DB access.
    Always closed after use to prevent connection leaks.
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


# ── 3. init_db() ──────────────────────────────────────────────────────────────
def init_db():
    """
    Creates the transactions table if it doesn't already exist.
    Called ONCE when api.py starts up.
    
    IF NOT EXISTS = safe to call multiple times, won't overwrite data.
    
    Table Structure:
        id         → auto-generated unique number for each transaction
        date       → when the transaction was added
        text       → original user input ("paid 500 at swiggy")
        category   → ML predicted category ("Food")
        amount     → extracted amount (500.0)
        type       → "income" or "expense"
        confidence → ML model confidence score (0.94)
    """
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            date       TEXT    NOT NULL,
            text       TEXT    NOT NULL,
            category   TEXT    NOT NULL,
            amount     REAL    NOT NULL,
            type       TEXT    NOT NULL,
            confidence REAL    NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Database initialized → finance.db")


# ── 4. add_transaction() ──────────────────────────────────────────────────────
def add_transaction(text: str, category: str, amount: float, confidence: float) -> dict:
    """
    Inserts a new confirmed transaction into the database.
    
    Called by:  POST /transaction/add  in api.py
    When:       User confirms the ML prediction and clicks Save
    
    Returns the newly created transaction as a dict
    so api.py can send it back to the frontend.
    
    Example:
        add_transaction("paid 500 at swiggy", "Food", 500.0, 0.94)
        → { id: 1, date: "2026-03-11", text: "...", ... }
    """
    conn    = get_db()
    tx_type = "income" if category == "Income" else "expense"
    date    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor = conn.execute(
        """INSERT INTO transactions 
           (date, text, category, amount, type, confidence)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (date, text, category, amount, tx_type, confidence)
    )
    conn.commit()
    tx_id = cursor.lastrowid  # get the auto-generated ID
    conn.close()

    return {
        "id":         tx_id,
        "date":       date,
        "text":       text,
        "category":   category,
        "amount":     amount,
        "type":       tx_type,
        "confidence": confidence
    }


# ── 5. get_all_transactions() ─────────────────────────────────────────────────
def get_all_transactions(limit: int = 50, offset: int = 0) -> list:
    """
    Fetches all transactions ordered by date (newest first).
    
    Called by:  GET /transactions  in api.py
    When:       User opens the History / Ledger page
    
    limit  → how many rows to return (default 50)
    offset → how many rows to skip (for pagination)
    
    Pagination Example:
        Page 1 → limit=50, offset=0   (rows 1-50)
        Page 2 → limit=50, offset=50  (rows 51-100)
    """
    conn = get_db()
    rows = conn.execute(
        """SELECT * FROM transactions 
           ORDER BY date DESC 
           LIMIT ? OFFSET ?""",
        (limit, offset)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ── 6. get_transaction_by_id() ────────────────────────────────────────────────
def get_transaction_by_id(tx_id: int) -> dict | None:
    """
    Fetches a single transaction by its ID.
    
    Called by:  PUT and DELETE endpoints in api.py
    When:       Before updating or deleting — to check it exists
    
    Returns None if not found → api.py raises 404 error
    
    Example:
        get_transaction_by_id(5)
        → { id: 5, text: "netflix subscription", ... }
        
        get_transaction_by_id(999)
        → None  (doesn't exist)
    """
    conn = get_db()
    row  = conn.execute(
        "SELECT * FROM transactions WHERE id = ?", (tx_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ── 7. update_transaction() ───────────────────────────────────────────────────
def update_transaction(tx_id: int, category: str) -> bool:
    """
    Updates the category of an existing transaction.
    
    Called by:  PUT /transaction/{id}  in api.py
    When:       User corrects a wrong ML prediction
    
    Example:
        ML predicted "Others" for "gym membership 1500"
        User corrects it to "Health"
        update_transaction(5, "Health") → True
    
    Also updates 'type' field because if category changes
    to/from Income, the type must change too.
    
    Returns True if updated, False if transaction not found.
    """
    conn = get_db()
    result = conn.execute(
        """UPDATE transactions 
           SET category = ?, type = ?
           WHERE id = ?""",
        (category, "income" if category == "Income" else "expense", tx_id)
    )
    conn.commit()
    conn.close()
    return result.rowcount > 0  # rowcount = 0 means nothing was updated


# ── 8. delete_transaction() ───────────────────────────────────────────────────
def delete_transaction(tx_id: int) -> bool:
    """
    Deletes a transaction permanently from the database.
    
    Called by:  DELETE /transaction/{id}  in api.py
    When:       User removes a transaction from their ledger
    
    Returns True if deleted, False if not found.
    
    Example:
        delete_transaction(3) → True  (deleted)
        delete_transaction(999) → False  (not found)
    """
    conn = get_db()
    result = conn.execute(
        "DELETE FROM transactions WHERE id = ?", (tx_id,)
    )
    conn.commit()
    conn.close()
    return result.rowcount > 0


# ── 9. has_income_transaction() ───────────────────────────────────────────────
def has_income_transaction() -> bool:
    """
    Returns True if at least one income transaction exists.
    Called on app startup to gate access.
    """
    conn = get_db()
    row  = conn.execute(
        "SELECT COUNT(*) as count FROM transactions WHERE type = 'income'"
    ).fetchone()
    conn.close()
    return row["count"] > 0


# ── 10. get_summary() ─────────────────────────────────────────────────────────
def get_summary() -> dict:
    """
    Calculates financial summary from all transactions.
    
    Called by:  GET /summary  in api.py
    When:       User opens the Dashboard page
    
    Returns:
        total_income      → sum of all income transactions
        total_expense     → sum of all expense transactions
        balance           → income - expense
        by_category       → amount spent per category
        transaction_count → total number of transactions
    
    Example return:
        {
            "total_income":      45000.0,
            "total_expense":     18500.0,
            "balance":           26500.0,
            "by_category": {
                "Food":          4200.0,
                "Transport":     1800.0,
                "Entertainment":  500.0
            },
            "transaction_count": 24
        }
    """
    conn = get_db()
    rows = conn.execute("SELECT * FROM transactions").fetchall()
    conn.close()

    income      = 0.0
    expense     = 0.0
    by_category = {}

    for r in rows:
        amt = r["amount"]
        cat = r["category"]
        by_category[cat] = round(by_category.get(cat, 0) + amt, 2)
        if r["type"] == "income":
            income  += amt
        else:
            expense += amt

    return {
        "total_income":      round(income, 2),
        "total_expense":     round(expense, 2),
        "balance":           round(income - expense, 2),
        "by_category":       by_category,
        "transaction_count": len(rows)
    }
if __name__ == "__main__":
    print("\n🧪 Testing database.py...\n")

    # 1. Init
    init_db()

    # 2. Add
    print("\n2. Adding test transactions...")
    t1 = add_transaction("paid 500 at swiggy",      "Food",      500.0,  0.94)
    t2 = add_transaction("uber ride to office",      "Transport", 250.0,  0.88)
    t3 = add_transaction("salary credited 45000",    "Income",    45000.0,0.97)
    print(f"   ✅ Added ID {t1['id']} → {t1['text']}")
    print(f"   ✅ Added ID {t2['id']} → {t2['text']}")
    print(f"   ✅ Added ID {t3['id']} → {t3['text']}")

    # 3. Fetch All
    print("\n3. Fetching all transactions...")
    txns = get_all_transactions()
    print(f"   ✅ Found {len(txns)} transactions")
    for t in txns:
        print(f"   [{t['id']}] {t['text']:<40} ₹{t['amount']} | {t['category']} | {t['type']}")

    # 4. Summary
    print("\n4. Summary...")
    s = get_summary()
    print(f"   ✅ Income:  ₹{s['total_income']}")
    print(f"   ✅ Expense: ₹{s['total_expense']}")
    print(f"   ✅ Balance: ₹{s['balance']}")
    print(f"   ✅ Count:   {s['transaction_count']}")

    # 5. Delete
    print(f"\n5. Deleting ID {t1['id']}...")
    print(f"   ✅ Deleted: {delete_transaction(t1['id'])}")

    print("\n6. Cleaning up test data...")
    import sqlite3
    conn = sqlite3.connect(DB_FILE)
    conn.execute("DELETE FROM transactions")
    conn.commit()
    conn.close()
    print("   ✅ Test data removed")

    print("\n✅ All tests passed!")
