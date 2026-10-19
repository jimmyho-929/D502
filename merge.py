import pandas as pd
import hashlib

# Load files
item_details = pd.read_csv("datasets/2023/2023_All_Items.csv")
order_details = pd.read_csv("datasets/2023/2023_All_Orders.csv")

# Parse timestamps
item_details["Sent Date"] = pd.to_datetime(item_details["Sent Date"], format="%m/%d/%y %H:%M")
order_details["Opened"] = pd.to_datetime(order_details["Opened"], format="%m/%d/%y %H:%M")


# Extract just the date part (no time) for the composite key
order_details["date"] = order_details["Opened"].dt.date
item_details["date"] = item_details["Sent Date"].dt.date

# Build composite key: date + order number
order_details["order_key"] = order_details["date"].astype(str) + "_" + order_details["Order #"].astype(str)
item_details["order_key"] = item_details["date"].astype(str) + "_" + item_details["Order #"].astype(str)

# Merge on the composite key
merged = item_details.merge(
    order_details.drop(columns=["date"]),
    on="order_key",
    how="left",
    suffixes=("_item", "_order")
)

# Removes rows where the merge failed (i.e., no matching order found for an item)
merged = merged.dropna(subset=["Opened"])

# Clean up duplicate Order # columns from the merge
merged = merged.drop(columns=["Order #_order"])
merged = merged.rename(columns={"Order #_item": "Order #"})

# Drop voided orders
merged = merged[merged["Void?"] == False]
merged = merged.drop(columns=["Void?"])

# Pseudonymize Server
def pseudonymize(name):
    return "Server_" + hashlib.sha256(str(name).encode()).hexdigest()[:6]

merged["Server"] = merged["Server"].apply(lambda x: x if x == "Online Ordering" else pseudonymize(x))

# Fill missing Table values 
merged["Table"] = merged["Table"].fillna("N/A")

# Collapse anything with "(GH)" into "GrubHub"
merged["Menu"] = merged["Menu"].apply(lambda x: "GrubHub" if "(GH)" in str(x) else x)

# Fix Sales Category — beverages mislabeled as "Food"
merged.loc[merged["Menu Group"].str.contains("Beverage", case=False, na=False), "Sales Category"] = "Beverage"
merged["Sales Category"] = merged["Sales Category"].replace("N/A Beverages", "Beverage")

# Merge Gratuity into Tip, then drop Gratuity
merged["Tip"] = merged["Tip"] + merged["Gratuity"]
merged = merged.drop(columns=["Gratuity"])

# Build Channel: Delivery (GrubHub) / In-House / Online
def get_channel(row):
    if "GrubHub" in str(row["Menu"]):
        return "Delivery"
    elif row["Server"] != "Online Ordering":
        return "In-House"
    else:
        return "Online"

merged["Channel"] = merged.apply(get_channel, axis=1)

# Extract Day of Week, Hour, Meal Period from Opened
merged["Day of Week"] = merged["Opened"].dt.day_name()
merged["Hour"] = merged["Opened"].dt.hour

# Lunch/Dinner cutoff at 3 PM
def meal_period(hour):
    if hour < 15:
        return "Lunch"
    else:
        return "Dinner"

merged["Meal Period"] = merged["Hour"].apply(meal_period)

# Final dtype casting
merged["Qty"] = merged["Qty"].astype(int)
merged["# of Guests"] = merged["# of Guests"].astype(int)
merged["Hour"] = merged["Hour"].astype(int)

# Reorder Order Key to second column
cols = merged.columns.tolist()
cols.remove("order_key")
cols.insert(1, "order_key")

# Final cleanup and save
merged = merged[cols]
merged = merged.rename(columns={"order_key": "Order Key"})

merged = merged.dropna(subset=["Menu Group", "Menu", "Sales Category"])
merged = merged.drop(columns=["date","Sent Date"])

merged.to_csv("datasets/2023_merged_data.csv", index=False)
