import pandas as pd

merged = pd.read_csv("datasets/22_23_merged_data.csv", parse_dates=["Opened"])

# Start from cleaned merged dataset
# Dropping duplicates on Order Key leaves one row per order, so counts and sums are not inflated.
order_level = merged.drop_duplicates(subset="Order Key").copy()
order_level["Order Date"] = pd.to_datetime(order_level["Order Date"])

# Aggregate to one row per day
daily = order_level.groupby("Order Date").agg(
    order_count=("Order Key", "count"),
    total_guests=("# of Guests", "sum"),
    avg_guests=("# of Guests", "mean"),
    total_amount=("Amount", "sum"),
    avg_amount=("Amount", "mean"),
    total_tip=("Tip", "sum"),
).reset_index()

# Time-based features
daily["Day of Week"] = daily["Order Date"].dt.day_name()
daily["Month"] = daily["Order Date"].dt.month
daily["Year"] = daily["Order Date"].dt.year
daily["Is Weekend"] = daily["Day of Week"].isin(["Friday", "Saturday"]).astype(int)
daily["Day of Year"] = daily["Order Date"].dt.dayofyear
daily["Week of Year"] = daily["Order Date"].dt.isocalendar().week

# Sanity check for closure days (near-zero order volume) across both years
print("Potential closure days (order_count < 50):")
print(daily[daily["order_count"] < 50][["Order Date", "order_count"]])

# Drop closure days if present (adjust threshold/list based on the printed output above)
closures = daily[daily["order_count"] < 50]["Order Date"]
daily = daily[~daily["Order Date"].isin(closures)]

# Lag / rolling features
daily = daily.sort_values("Order Date")
daily["lag_1"] = daily["order_count"].shift(1)
daily["lag_7"] = daily["order_count"].shift(7)
daily["rolling_7_avg"] = daily["order_count"].shift(1).rolling(7).mean()

# Sanity check the year-boundary transition
print("Year-boundary check (Dec 20 - Jan 10):")
print(daily[(daily["Order Date"] >= "2022-12-20") & (daily["Order Date"] <= "2023-01-10")]
      [["Order Date", "order_count", "lag_1", "lag_7"]])

# Channel mix as features
channel_daily = order_level.groupby(["Order Date", "Channel"])["Order Key"].count().unstack(fill_value=0)
channel_daily.columns = [f"{c}_orders" for c in channel_daily.columns]
daily = daily.merge(channel_daily, on="Order Date", how="left")

# Turns the day name into 0/1 columns. drop_first=True removes one category
# (Friday) to avoid perfect collinearity; Friday becomes the reference day, so
# the Linear Regression coefficients read as "orders vs. Friday".
daily = pd.get_dummies(daily, columns=["Day of Week"], drop_first=True)

# Drop rows with NaN from lag features (first week)
daily = daily.dropna(subset=["lag_1", "lag_7", "rolling_7_avg"])

# Convert True/False dummies to 0/1 and the nullable-integer week to a normal int
# so scikit-learn handles every column consistently.
bool_cols = daily.select_dtypes(include="bool").columns
daily[bool_cols] = daily[bool_cols].astype(int)
daily["Week of Year"] = daily["Week of Year"].astype(int)


#Define the leak-free feature list (for reference/printing)
leaky_cols = [
    "total_amount", "total_guests", "avg_amount", "avg_guests", "total_tip",
    "Delivery_orders", "In-House_orders", "Online_orders"
]

feature_cols = [c for c in daily.columns if c not in ["Order Date", "order_count"] + leaky_cols]

X = daily[feature_cols]
y = daily["order_count"]

print(daily.shape)
print(X.columns.tolist())

daily.to_csv("datasets/daily_features.csv", index=False)
