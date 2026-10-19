import pandas as pd

# One row per operating day. Sorting by date is essential here because the
# split below is chronological, and the lag/rolling features assume time order.
daily = pd.read_csv("datasets/daily_features.csv", parse_dates=["Order Date"])
daily = daily.sort_values("Order Date").reset_index(drop=True)


# Leaky columns are same-day aggregates (totals, guest counts, channel counts).
# They are strongly correlated with the target (order_count) and would not be
# known in advance in a real forecast, so including them would inflate accuracy.
leaky_cols = [
    "total_amount", "total_guests", "avg_amount", "avg_guests", "total_tip",
    "Delivery_orders", "In-House_orders", "Online_orders"
]

# Columns excluded from the model:
#   Order Date   - identifier, not a numeric predictor (kept in daily for plotting)
#   order_count  - the target we are predicting
#   Is Weekend, Day of Year, Week of Year - redundant with Day of Week / Month
#                  and caused multicollinearity that distorted the coefficients
drop_cols = ["Order Date", "order_count", "Is Weekend", "Day of Year", "Week of Year"] + leaky_cols

feature_cols = [c for c in daily.columns if c not in drop_cols]

X = daily[feature_cols]
y = daily["order_count"]

# Chronological train/test split
cutoff_date = "2023-10-01"
train_mask = daily["Order Date"] < cutoff_date
test_mask = daily["Order Date"] >= cutoff_date

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]


# Quick checks: confirm sizes and the feature list
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Features used:", feature_cols)

# Save the splits
X_train.to_csv("datasets/X_train.csv", index=False)
X_test.to_csv("datasets/X_test.csv", index=False)
y_train.to_csv("datasets/y_train.csv", index=False)
y_test.to_csv("datasets/y_test.csv", index=False)