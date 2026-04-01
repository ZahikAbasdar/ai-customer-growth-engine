import pandas as pd
import numpy as np

# ⚙️ CONFIG (change size here)
ROWS = 200000  # increase to 500k or 1M if you want bigger

print("Generating data...")

# Generate synthetic Amazon-like data
data = {
    "product_id": np.random.randint(100000, 999999, ROWS),
    "user_id": np.random.randint(10000, 99999, ROWS),
    "price": np.round(np.random.uniform(100, 5000, ROWS), 2),
    "rating": np.random.randint(1, 6, ROWS),
    "review_length": np.random.randint(20, 500, ROWS),
    "purchase_frequency": np.random.randint(1, 50, ROWS),
    "discount": np.round(np.random.uniform(0, 50, ROWS), 2),
    "delivery_time": np.random.randint(1, 10, ROWS),
    "stock_left": np.random.randint(0, 1000, ROWS),
    "wishlist_count": np.random.randint(0, 500, ROWS)
}

df = pd.DataFrame(data)

# Save file
filename = "big_amazon_data.csv"
df.to_csv(filename, index=False)

print(f"✅ Data saved as {filename}")
print(f"Rows: {len(df)}")