from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys

sys.stdout.reconfigure(encoding='utf-8')

try:
    from inference import predict_transaction
except ImportError as e:
    raise RuntimeError(f"Failed to load inference module: {e}")

from database import (
    init_db,
    add_transaction,
    get_all_transactions,
    get_transaction_by_id,
    update_transaction,
    delete_transaction,
    get_summary,
    has_income_transaction
)

app = FastAPI(
    title="Finance Tracker API",
    description="API for categorizing financial transactions and extracting amounts.",
    version="1.0.0"
)

init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    text:       str
    category:   str
    confidence: float
    status:     str
    amount:     Optional[float]
    amount_str: str
    type:       str

class AddTransactionRequest(BaseModel):
    text:       str
    category:   str
    amount:     float
    confidence: float

class TransactionResponse(BaseModel):
    id:         int
    date:       str
    text:       str
    category:   str
    amount:     float
    type:       str
    confidence: float

class UpdateCategoryRequest(BaseModel):
    category: str

class SummaryResponse(BaseModel):
    total_income:      float
    total_expense:     float
    balance:           float
    by_category:       dict
    transaction_count: int

@app.get("/")
def read_root():
    return {
        "message": "Finance Tracker API is live",
        "status":  "active",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Transaction text cannot be empty.")
    try:
        cat, conf, amount = predict_transaction(request.text)
        if conf >= 0.80:
            status = "High Confidence"
        elif conf >= 0.50:
            status = "Medium Confidence"
        else:
            status = "Low Confidence"
        return PredictResponse(
            text=request.text,
            category=cat,
            confidence=round(float(conf), 4),
            status=status,
            amount=amount,
            amount_str=f"₹{amount}" if amount is not None else "Not found",
            type="income" if cat == "Income" else "expense"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transaction/add", response_model=TransactionResponse)
def add(req: AddTransactionRequest):
    if req.category != "Income" and not has_income_transaction():
        raise HTTPException(
            status_code=400,
            detail="Please add your income first before logging expenses."
        )
    try:
        result = add_transaction(
            text=req.text,
            category=req.category,
            amount=req.amount,
            confidence=req.confidence
        )
        return TransactionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transactions", response_model=list[TransactionResponse])
def get_transactions(limit: int = 50, offset: int = 0):
    try:
        rows = get_all_transactions(limit=limit, offset=offset)
        return [TransactionResponse(**row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transaction/{tx_id}", response_model=TransactionResponse)
def get_transaction(tx_id: int):
    row = get_transaction_by_id(tx_id)
    if not row:
        raise HTTPException(status_code=404, detail="Transaction not found.")
    return TransactionResponse(**row)

@app.get("/summary", response_model=SummaryResponse)
def summary():
    try:
        result = get_summary()
        return SummaryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/transaction/{tx_id}")
def update(tx_id: int, req: UpdateCategoryRequest):
    row = get_transaction_by_id(tx_id)
    if not row:
        raise HTTPException(status_code=404, detail="Transaction not found.")
    success = update_transaction(tx_id=tx_id, category=req.category)
    if not success:
        raise HTTPException(status_code=500, detail="Update failed.")
    return {
        "message":  "Category updated successfully",
        "id":       tx_id,
        "category": req.category
    }

@app.delete("/transaction/{tx_id}")
def delete(tx_id: int):
    row = get_transaction_by_id(tx_id)
    if not row:
        raise HTTPException(status_code=404, detail="Transaction not found.")
    success = delete_transaction(tx_id)
    if not success:
        raise HTTPException(status_code=500, detail="Delete failed.")
    return {
        "message": "Transaction deleted successfully",
        "id":      tx_id
    }

@app.get("/has-income")
def check_has_income():
    try:
        has = has_income_transaction()
        return {"has_income": has}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting Finance Tracker API → http://127.0.0.1:8000")
    print("API Docs        → http://127.0.0.1:8000/docs")
    print("Alt Docs        → http://127.0.0.1:8000/redoc")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)