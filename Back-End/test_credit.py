import sys
sys.stdout.reconfigure(encoding='utf-8')
from inference import predict_transaction

tests = [
    "Credit 5000 received from client",
    "credited 10000 to my account",
    "Bank credit 25000",
    "salary credited 45000",
    "cashback credited 200",
    "received payment 15000",
    "income from freelance 8000",
    "refund 500 credited",
    "UPI credit 3000 from friend",
    "money received 2000",
]

print(f"{'Input':<45} {'Category':<15} {'Conf':<8} Amount")
print("-" * 85)
for t in tests:
    cat, conf, amt = predict_transaction(t)
    amount_str = f"Rs.{amt}" if amt else "Not found"
    flag = "YES" if cat == "Income" else "no"
    print(f"{t:<45} {cat:<15} {conf:<8.2f} {amount_str}  [{flag}]")
