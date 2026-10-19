import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)
# X = feature columns, y = target (daily order count).
# The split is chronological: train = Jan 2022 to Sep 2023, test = Oct to Dec 2023.
# .squeeze() converts the single-column y DataFrames back into Series.
X_train = pd.read_csv("datasets/X_train.csv")
X_test = pd.read_csv("datasets/X_test.csv")
y_train = pd.read_csv("datasets/y_train.csv").squeeze()
y_test = pd.read_csv("datasets/y_test.csv").squeeze()


# Linear Regression: interpretable; coefficients show each feature's effect.
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

# Random Forest : averages many decision trees to capture non-linear patterns.
# random_state=42 makes results reproducible (also used in visualizations.py).
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# Evaluate models on the test set
# Metrics:
#   RMSE - typical error in orders/day; penalizes large misses more heavily
#   MAE  - average absolute error in orders/day; easier to interpret
#   MAPE - average error as a percentage of actual volume
#   R2   - share of variance in daily orders explained (1.0 = perfect;
#          0 = no better than predicting the average; negative = worse)
results = {}
for name, preds in [("Linear Regression", lr_preds), ("Random Forest", rf_preds)]:
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results[name] = {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}
    print(f"{name} — RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2%}, R²: {r2:.4f}")

# Baseline rule: "predict the same count as 7 days ago" (the lag_7 feature).
naive_preds = X_test["lag_7"]
naive_rmse = np.sqrt(mean_squared_error(y_test, naive_preds))
naive_mae = mean_absolute_error(y_test, naive_preds)
naive_mape = mean_absolute_percentage_error(y_test, naive_preds)
naive_r2 = r2_score(y_test, naive_preds)
print(f"Naive (lag_7) baseline — RMSE: {naive_rmse:.2f}, MAE: {naive_mae:.2f}, MAPE: {naive_mape:.2%}, R²: {naive_r2:.4f}")

# Save predictions for visualization step
predictions = pd.DataFrame({
    "y_test": y_test,
    "lr_preds": lr_preds,
    "rf_preds": rf_preds,
    "naive_preds": naive_preds
})
predictions.to_csv("datasets/model_predictions.csv", index=False)

# Save full results table for reporting
results_df = pd.DataFrame(results).T
results_df.loc["Naive Baseline"] = [naive_rmse, naive_mae, naive_mape, naive_r2]
results_df.to_csv("datasets/model_results.csv")
print(results_df)