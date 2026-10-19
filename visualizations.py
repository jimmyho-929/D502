import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

# Load train/test data
# X = feature columns, y = target (daily order count).
# .squeeze() converts the single-column y DataFrames back into Series.
X_train = pd.read_csv("datasets/X_train.csv")
X_test = pd.read_csv("datasets/X_test.csv")
y_train = pd.read_csv("datasets/y_train.csv").squeeze()
y_test = pd.read_csv("datasets/y_test.csv").squeeze()

# Retrain both models (quick, since dataset is small)
# This script runs separately from model_training.py, so the trained
# models are not in memory. Retraining is fast on a dataset this small.
# random_state=42 keeps the Random Forest results reproducible and
# identical to those from model_training.py.
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

lr = LinearRegression()
lr.fit(X_train, y_train)

daily = pd.read_csv("datasets/daily_features.csv", parse_dates=["Order Date"])
predictions = pd.read_csv("datasets/model_predictions.csv")

cutoff_date = "2023-10-01"
test_dates = daily[daily["Order Date"] >= cutoff_date]["Order Date"].reset_index(drop=True)
predictions["Order Date"] = test_dates

# Chart 1: Actual vs Predicted
# Headline visual: shows how closely each model tracks real demand.
plt.figure(figsize=(14, 6))
plt.plot(predictions["Order Date"], predictions["y_test"], label="Actual", linewidth=2)
plt.plot(predictions["Order Date"], predictions["lr_preds"], label="Linear Regression", linestyle="--")
plt.plot(predictions["Order Date"], predictions["rf_preds"], label="Random Forest", linestyle="--")
plt.title("Actual vs. Predicted Daily Order Volume (Oct–Dec 2023)")
plt.xlabel("Date")
plt.ylabel("Order Count")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

# Chart 2: Residuals over time
# Residual = actual - predicted.
#   Positive bar = model under-predicted (actual was higher)
#   Negative bar = model over-predicted (actual was lower)
# Used to check for bias or drift and to spot outliers
predictions["error"] = predictions["y_test"] - predictions["lr_preds"]

plt.figure(figsize=(14, 5))
plt.bar(predictions["Order Date"], predictions["error"])
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Linear Regression Prediction Error Over Time")
plt.xlabel("Date")
plt.ylabel("Error (Actual - Predicted)")
plt.tight_layout()
plt.savefig("residuals_over_time.png")
plt.show()

# Chart 3: Random Forest feature importance
# Importances sum to 1.0; a higher value means the feature contributed more
# to the forest's splitting decisions. Sorted ascending so the most important
# feature appears at the top of the horizontal bar chart.

importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values()

plt.figure(figsize=(8, 6))
importances.plot(kind="barh")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# Chart 4: Model comparison (RMSE / MAPE / MAE/ R²)
# Pull the predictions for each model out of the saved file.
lr_preds = predictions["lr_preds"]
rf_preds = predictions["rf_preds"]
naive_preds = X_test["lag_7"] 

models = ["Linear Regression", "Random Forest", "Naive Baseline"]
all_preds = [lr_preds, rf_preds, naive_preds]

results_df = pd.DataFrame({
    "Model": models,
    "RMSE": [np.sqrt(mean_squared_error(y_test, p)) for p in all_preds],
    "MAE":  [mean_absolute_error(y_test, p) for p in all_preds],
    "MAPE": [mean_absolute_percentage_error(y_test, p) * 100 for p in all_preds],
    "R2":   [r2_score(y_test, p) for p in all_preds],
})
print(results_df.round(3))  # quick check against model_training.py output

metric_info = [
    ("RMSE", "RMSE (lower is better)", "{:.1f}", "steelblue"),
    ("MAE",  "MAE (lower is better)",  "{:.1f}", "darkorange"),
    ("MAPE", "MAPE % (lower is better)", "{:.1f}%", "firebrick"),
    ("R2",   "R² (higher is better)",  "{:.3f}", "seagreen"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, (col, title, fmt, color) in zip(axes.flatten(), metric_info):
    bars = ax.bar(results_df["Model"], results_df[col], color=color, alpha=0.7)
    # Exact values on each bar so readers don't have to estimate from the axis
    ax.bar_label(bars, labels=[fmt.format(v) for v in results_df[col]], padding=3)
    ax.set_title(title)
    ax.margins(y=0.15)  # headroom so labels aren't clipped at the top

# R² can be negative, so a zero line is drawn to show clearly that a negative score is worse than predicting the average.
axes[1, 1].axhline(0, color="black", linewidth=0.8)

fig.suptitle("Model Comparison: RMSE, MAE, MAPE, and R²", fontsize=14)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()