import pandas as pd
import matplotlib.pyplot as plt

merged = pd.read_csv("datasets/22_23_merged_data.csv", parse_dates=["Opened"])

# One row per unique order (since order_key repeats across item rows)
daily_orders = (
    merged.drop_duplicates(subset="Order Key")
    .groupby(merged.drop_duplicates(subset="Order Key")["Opened"].dt.date)["Order Key"]
    .count()
)

daily_orders.index = pd.to_datetime(daily_orders.index)

# Identify closure days (near-zero order volume)
closures = daily_orders[daily_orders < 50]
print(closures)

closure_dates = closures.index

# Drop those dates from the full merged dataset
merged["Order Date"] = pd.to_datetime(merged["Opened"]).dt.date
merged["Order Date"] = pd.to_datetime(merged["Order Date"])

merged = merged[~merged["Order Date"].isin(closure_dates)]


# Chart 1: Daily order volume with closures removed
# Shows trend and weekly seasonality. The near-zero drops should be gone.
daily_orders = (
    merged.drop_duplicates(subset="Order Key")
    .groupby(merged.drop_duplicates(subset="Order Key")["Opened"].dt.date)["Order Key"]
    .count()
)

merged.to_csv("datasets/222_23_merged_data.csv", index=False)

daily_orders.index = pd.to_datetime(daily_orders.index)

plt.figure(figsize=(16, 5))
plt.plot(daily_orders.index, daily_orders.values)
plt.title("Daily Order Volume — '22 + '23")
plt.xlabel("Date")
plt.ylabel("Number of Orders")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("daily_order_volume_cleaned.png")
plt.show()

# Chart 2: Orders by day of week
merged["Order Date"] = pd.to_datetime(merged["Order Date"])

order_level = merged.drop_duplicates(subset="Order Key")
order_level.groupby("Day of Week")["Order Key"].count().reindex(
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
).plot(kind="bar", title="Orders by Day of Week", figsize=(8,5))
plt.tight_layout()
plt.show()

# Chart 3: Orders by hour of day
order_level.groupby("Hour")["Order Key"].count().plot(kind="bar", title="Orders by Hour", figsize=(10,5))
plt.tight_layout()
plt.show()


# Chart 4: Orders by day of week, split by channel
order_level.groupby(["Day of Week", "Channel"])["Order Key"].count().unstack().reindex(
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
).plot(kind="bar", stacked=True, figsize=(10,6), title="Orders by Day of Week and Channel")
plt.tight_layout()
plt.show()


# Chart 5: Top 20 menu items by quantity sold
top_items = merged.groupby("Menu Item")["Qty"].sum().sort_values(ascending=False).head(20)
print(top_items)

top_items.plot(kind="barh", figsize=(8,8), title="Top 20 Menu Items by Quantity Sold")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Chart 6: Orders by month
# Checks for seasonal effects. to_period("M") groups by year-month (e.g., 2022-01).
monthly_orders = order_level.groupby(order_level["Order Date"].dt.to_period("M"))["Order Key"].count()
monthly_orders.plot(kind="bar", title="Orders by Month", figsize=(10,5))
plt.tight_layout()
plt.show()

# Chart 7: Average order amount by day of week
order_level["Amount"].describe()

order_level.groupby("Day of Week")["Amount"].mean().reindex(
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
).plot(kind="bar", title="Average Order Amount by Day of Week", figsize=(8,5))
plt.tight_layout()
plt.show()


# Chart 8: Pad Thai demand by hour
pad_thai = merged[merged["Menu Item"] == "Pad Thai"]
pad_thai.groupby("Hour")["Qty"].sum().plot(kind="bar", title="Pad Thai Orders by Hour")
plt.tight_layout()
plt.show()