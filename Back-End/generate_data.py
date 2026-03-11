import csv
import random
import argparse
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Configurable Parameters
TOTAL_SAMPLES = 3000
RANDOM_SEED = 42
IMBALANCE = True
normalize_amounts_flag = False # Will be set by args

random.seed(RANDOM_SEED)

# 1. Categories & Merchants / Keywords
categories = {
    "Food": {
        "merchants": ["Swiggy", "Zomato", "Dominos", "Pizza Hut", "McDonalds", "Burger King", "Starbucks", "Subway", "KFC", "Blinkit", "BigBasket", "Instamart"],
        "keywords": ["lunch", "dinner", "breakfast", "snack", "coffee", "grocery", "vegetables", "fruits", "restaurant", "cafe", "food"],
        "freeform": ["went out for dinner with friends", "ordered late night snacks", "bought vegetables from local shop", "tea break at stall", "office canteen lunch", "sunday brunch bill", "late night craving order", "birthday treat for team"]
    },
    "Transport": {
        "merchants": ["Uber", "Ola", "Rapido", "Indigo", "Air Asia", "IRCTC", "Shell", "HP Petrol"],
        "keywords": ["ride", "trip", "cab", "auto", "flight", "train", "bus", "fuel", "petrol", "diesel", "parking", "toll", "metro"],
        "freeform": ["office cab ride", "fuel filled for bike", "metro ride to work", "booked flight tickets for vacation", "train ticket booking tatkal", "auto rickshaw fare", "monthly bus pass renewal", "parking charges at mall"]
    },
    "Housing": {
        "merchants": ["Urban Company", "NoBroker", "IKEA", "Home Centre", "Asian Paints"],
        "keywords": ["rent", "maintenance", "electricity", "water bill", "gas bill", "plumber", "electrician", "maid", "carpenter", "repairs", "furniture"],
        "freeform": ["paid monthly house rent", "electrician repair work", "maid salary for this month", "bought new curtains for hall", "plumbing service charge", "society maintenance paid", "whitewashing labor cost"]
    },
    "Entertainment": {
        "merchants": ["Netflix", "Spotify", "Amazon Prime", "BookMyShow", "PVR", "IMAX", "Disney+", "Steam", "PlayStation"],
        "keywords": ["movie", "cinema", "subscription", "game", "concert", "event", "show", "party"],
        "freeform": ["movie night with family", "concert tickets booking", "renewed netflix subscription", "bought new ps5 game", "weekend party expenses", "bowling alley bill", "streaming service monthly fee"]
    },
    "Shopping": {
        "merchants": ["Amazon", "Flipkart", "Myntra", "Zara", "H&M", "Nike", "Adidas", "Ajio", "Decathlon", "Apple Store"],
        "keywords": ["clothes", "shoes", "electronics", "gift", "shopping", "purchase", "apparel", "accessories"],
        "freeform": ["bought new sneakers", "diwali shopping for family", "gift for anniversary", "ordered clothes online", "electronics store bill", "winter wear purchase", "shopping spree at mall"]
    },
    "Utilities": {
        "merchants": ["Jio", "Airtel", "Vi", "Bescom", "Tata Sky", "DishTV", "Act Fibernet"],
        "keywords": ["recharge", "bill", "broadband", "internet", "dth", "mobile", "postpaid", "prepaid"],
        "freeform": ["mobile recharge done", "wifi bill payment", "dth yearly subscription", "electricity bill paid online", "gas cylinder booking", "water tanker charges", "internet connection setup fee"]
    },
    "Health": {
        "merchants": ["Apollo Pharmacy", "Practo", "1mg", "MedPlus", "Cure.fit", "Cult.fit"],
        "keywords": ["medicine", "doctor", "consultation", "test", "checkup", "gym", "fitness", "yoga", "pharmacy", "hospital", "clinic"],
        "freeform": ["doctor consultation fee", "bought medicines for flu", "gym membership renewal", "routine body checkup", "dental cleaning charges", "physiotherapy session", "vitamin supplements purchase"]
    },
    "Education": {
        "merchants": ["Udemy", "Coursera", "Kindle", "Byju's", "Unacademy"],
        "keywords": ["course", "book", "fee", "tuition", "school", "college", "workshop", "class", "training"],
        "freeform": ["semester fee payment", "bought clean coding book", "online python course", "school bus fee", "stationery connection", "exam registration fees", "workshop on ai"]
    },
    "Income": {
        "merchants": ["Employer", "Client", "Bank", "Google Pay", "PhonePe", "HDFC", "Friend"],
        "keywords": [
            "salary", "stipend", "bonus", "interest", "refund", "cashback",
            "dividend", "credited", "received", "credit", "income", "earnings",
            "payment received", "money received", "transferred", "deposited"
        ],
        "freeform": [
            "salary credited for august",
            "freelance payment received",
            "interest credited to account",
            "tax refund processed",
            "bonus received",
            "cashback from credit card",
            "received payment from client",
            "money received from friend",
            "UPI credit from friend",
            "payment received for project",
            "amount credited to account",
            "bank credit received",
            "income from freelance work",
            "money transferred to account",
            "received salary this month",
            "got paid for consulting",
            "transfer received from employer",
            "credit received in account",
            "amount deposited in bank",
            "received cashback on purchase",
            "annual bonus credited",
            "dividend credited to account",
            "reimbursement received from office",
            "rent received from tenant",
            "payment got credited"
        ]
    },
    "Others": {
        "merchants": ["Friend", "Charity", "Unknown"],
        "keywords": ["donation", "gift", "transfer", "withdrawal", "atm", "sent", "received", "misc"],
        "freeform": ["money sent to friend", "charity donation", "atm cash withdrawal", "unknown transaction", "petty cash expense", "gift for wedding", "loan repayment to friend"]
    },
    "Investment": {
        "merchants": ["Zerodha", "Groww", "Upstox", "Coin", "Binance", "WazirX", "CoinSwitch", "Mutual Fund", "SBI Mutual Fund"],
        "keywords": ["stocks", "sip", "crypto", "equity", "shares", "bonds", "invested", "investment", "portfolio", "deposit", "fd", "fixed deposit", "recurring deposit"],
        "freeform": ["invested in stocks", "monthly sip deduction", "bought bitcoin", "invested in mutual fund", "opened fixed deposit", "bought shares of company", "added money to zerodha wallet", "crypto purchase successful"]
    }
}

# 2. Templates
merchant_templates = [
    "Paid {amount} at {merchant}",
    "{amount} paid to {merchant}",
    "Transaction at {merchant} for {amount}",
    "{merchant} payment {amount}",
    "Spent {amount} on {merchant}",
    "Ordered from {merchant} {amount}",
    "Bill from {merchant} of {amount}",
    "{merchant} {amount}",
    "UPI payment to {merchant} {amount}",
    "Debit swipe {merchant} {amount}"
]

context_templates = [
    "{keyword} expense {amount}",
    "Paid for {keyword} {amount}",
    "{keyword} cost {amount}",
    "Spent {amount} on {keyword}",
    "Monthly {keyword} {amount}",
    "{keyword} payment of {amount}",
    "{amount} for {keyword}",
    "Just paid {keyword} bill {amount}",
    "{keyword} charges {amount}"
]

# 3. Helper Functions
def get_random_amount():
    return random.randint(10, 50000)

def format_amount(amount):
    """Formats amount in various ways."""
    if normalize_amounts_flag:
        return "<amount>"
        
    formats = [
        str(amount),
        f"₹{amount}",
        f"INR {amount}",
        f"{amount} rs",
        f"Rs. {amount}",
        f"{amount}.00",
        f"INR{amount}"
    ]
    return random.choice(formats)

def add_noise(text):
    """Adds realistic financial noise."""
    if normalize_amounts_flag:
         # If normalizing, we don't want to mess up the <amount> token too much or add numbers that look like amounts
         pass

    # Casing (Reduce excessive uppercase)
    case_choice = random.choice(['lower', 'title', 'normal', 'normal', 'normal']) # Bias towards normal/title
    if case_choice == 'lower':
        text = text.lower()
    elif case_choice == 'title':
        text = text.title()
    
    # Financial Fillers
    financial_fillers = [
        f" upi ref {random.randint(10000, 99999)}",
        f" txn id {random.randint(100000, 999999)}",
        " via gpay",
        " via phonepe",
        " via paytm",
        " successful",
        " debt",
        " cred",
        " on card"
    ]
    
    # Add filler with low probability < 20%
    if random.random() < 0.20:
        text += random.choice(financial_fillers)
        
    # Date phrases
    date_phrases = [" yesterday", " today", " last night", " on 12/05"]
    if random.random() < 0.10:
        text += random.choice(date_phrases)
        
    return text

def apply_ambiguity(category):
    """Introduces controlled cross-category ambiguity."""
    # 5% chance to switch category (e.g. Gym -> Entertainment instead of Health).
    # This helps the model learn decision boundaries and avoid overfitting to specific keywords.
    if random.random() > 0.05:
        return category
        
    if category == "Health":
        return "Entertainment" if random.random() < 0.5 else category # Gym -> Entertainment
    elif category == "Utilities":
        return "Housing" if random.random() < 0.5 else category # Broadband -> Housing
    elif category == "Others":
        return "Shopping" if random.random() < 0.5 else category # Gift -> Shopping
    
    return category

# Correction Templates for Amount Ambiguity
correction_templates = [
    "Paid {wrong} wait no {correct}",
    "Bill is {wrong}... sorry {correct}",
    "Transfer {wrong} actually {correct}",
    "Spent {wrong} oh wait it was {correct}",
    "{wrong} was debited... correction {correct}",
    "{correct} instead of {wrong}", # First number {correct} is true amount. "Paid 500 instead of 600".
    "Typo {wrong} read as {correct}",
    "{wrong} no make it {correct}"
]

# Negation Templates (True amount is NOT the one mentioned with 'not')
negation_templates = [
    "{correct} ... not {wrong}",
    "Bill is {correct} not {wrong}",
    "{correct} is the amount, not {wrong}",
    "Not {wrong} but {correct}", # "Not 3200 but 3000"
    "Calculated {wrong} incorrectly, actual is {correct}"
]

def generate_sample(category):
    # Handle ambiguity
    target_category = apply_ambiguity(category)

    data_source = categories[category]
    keywords_pool = data_source["keywords"]
    merchants_pool = data_source["merchants"]
    freeform_pool = data_source["freeform"]

    # Overwrite pools for specific ambiguity scenarios
    if category == "Entertainment" and random.random() < 0.05:
        keywords_pool = ["gym", "fitness center"]
        merchants_pool = ["Gold's Gym", "Cult.fit"]
        freeform_pool = ["gym membership renewal"]
    elif category == "Housing" and random.random() < 0.05:
        keywords_pool = ["broadband", "wifi"]
        freeform_pool = ["wifi bill payment"]
    elif category == "Shopping" and random.random() < 0.05:
        keywords_pool = ["gift"]
        freeform_pool = ["birthday gift"]

    true_amount = get_random_amount()
    formatted_amount = format_amount(true_amount)
    
    # 15% Chance of Correction OR Negation (Multiple Numbers)
    if random.random() < 0.15:
        wrong_amount = get_random_amount()
        while wrong_amount == true_amount:
            wrong_amount = get_random_amount()
            
        # 50/50 split between Correction and Negation patterns
        if random.random() < 0.5:
            template = random.choice(correction_templates)
        else:
            template = random.choice(negation_templates)
            
        # Randomly choose if the text uses merchant or keyword context
        prefix = random.choice(merchants_pool + keywords_pool)
        
        full_text = f"{prefix} {template.format(wrong=wrong_amount, correct=formatted_amount)}"
        text = full_text
        
    # Decision: Template vs Free-form
    elif random.random() < 0.15: # 15% Free-form
        text = random.choice(freeform_pool)
        # Optionally append amount if not present
        if not any(char.isdigit() for char in text) and "<amount>" not in text:
            # Always append amount for training consistency
            text += f" {formatted_amount}"
    else:
        # Template based
        is_merchant_based = random.random() > 0.4
        if is_merchant_based and merchants_pool:
            entity = random.choice(merchants_pool)
            template = random.choice(merchant_templates)
        else:
            entity = random.choice(keywords_pool)
            template = random.choice(context_templates)
        text = template.format(merchant=entity, keyword=entity, amount=formatted_amount)

    text = add_noise(text)
    return text, category, true_amount

# 4. Main Generation Logic
def generate_dataset(n_samples=TOTAL_SAMPLES, imbalance=IMBALANCE):
    data = []
    
    # Define weights
    if imbalance:
        weights = {
            "Food": 0.25, "Shopping": 0.20, "Transport": 0.15, "Utilities": 0.10,
            "Entertainment": 0.08, "Housing": 0.07, "Health": 0.05, 
            "Education": 0.04, "Income": 0.03, "Others": 0.03
        }
    else:
        weights = {k: 1.0/len(categories) for k in categories}

    samples_per_category = {k: int(v * n_samples) for k, v in weights.items()}
    
    # Fill remainder
    current_total = sum(samples_per_category.values())
    if current_total < n_samples:
        samples_per_category["Food"] += (n_samples - current_total)
        
    for category, count in samples_per_category.items():
        for _ in range(count):
            text, cat, amt = generate_sample(category)
            data.append([text, cat, amt])
            
    random.shuffle(data)
    return data

# 5. Execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic transaction data.")
    parser.add_argument("--samples", type=int, default=TOTAL_SAMPLES, help="Total number of samples")
    parser.add_argument("--imbalance", action="store_true", default=IMBALANCE, help="Enable class imbalance")
    parser.add_argument("--normalize", action="store_true", help="Replace amounts with <amount> token")
    args = parser.parse_args()
    
    if args.samples < 1000:
        print("\n⚠️  WARNING: Generating fewer than 1000 samples.")
        print("    Data extraction models need more data. Consider using --samples 3000.")
        import time
        time.sleep(2)
    
    normalize_amounts_flag = args.normalize
    
    dataset = generate_dataset(n_samples=args.samples, imbalance=args.imbalance)
    
    output_file = 'transactions.csv'
    unique_data = []
    seen = set()
    
    # Improved Deduplication
    for text, cat, amt in dataset:
        # Normalize for deduplication check
        clean_key = text.lower().strip()
        clean_key = "".join(c for c in clean_key if c.isalnum() or c.isspace())
        
        if clean_key not in seen:
            unique_data.append([text, cat, amt])
            seen.add(clean_key)
            
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'category', 'amount'])
        writer.writerows(unique_data)
        
    print(f"Generated {len(unique_data)} unique samples in '{output_file}'")
    print(f"Configuration: Samples={args.samples}, Imbalance={args.imbalance}, Normalize={args.normalize}")
    print("\n--- Example Rows ---")
    for i in range(min(15, len(unique_data))):
        print(f"{unique_data[i][0]}  |  {unique_data[i][1]}")
