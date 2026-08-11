
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split      # split train/test
from sklearn.ensemble import RandomForestRegressor         # model: random forest
from sklearn.metrics import mean_absolute_error            # calculate MAE

df = pd.read_csv("hygdata_clean.csv")
print(f"Number of stars: {len(df)}")

# X is "feature" - which the model uses to predict. Only 1 column: ci (màu sắc)
# sklearn requires X to be double columns
# so we use df[["ci"]] with []
X = df[["ci"]]

# y is "target" - which the model needs to predict
y = df["absmag"]

# Keep "mag" &  "dist" to help compare the data train and real data (do NOT feed the model)
mag_all = df["mag"]
dist_true_all = df["dist"]

X_train, X_test, y_train, y_test, mag_train, mag_test, dist_train, dist_test = \
    train_test_split(X, y, mag_all, dist_true_all, test_size=0.2, random_state=42)

print(f"Number of stars for TRAIN: {len(X_train)}")
print(f"Number of stars for TEST: {len(X_test)}")

def evaluate_distance(absmag_predicted, mag_test, dist_test, model_name):
    """
    Using the formula from step 2 to calculate.
    """
    dist_predicted = np.power(10, (mag_test - absmag_predicted + 5) / 5)

    error_percent = np.abs(dist_predicted - dist_test) / dist_test * 100

    mae_absmag = mean_absolute_error(y_test, absmag_predicted)

    print(f"\n----- Model's results: {model_name} -----")
    print(f"MAE of absmag: {mae_absmag:.3f}")
    print(f"Error % of distance - median:     {error_percent.median():.2f} %")
    print(f"Error % trên distance - mean:     {error_percent.mean():.2f} %")

    return dist_predicted, error_percent


model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
absmag_pred_rf = model_rf.predict(X_test)

evaluate_distance(absmag_pred_rf, mag_test, dist_test, "Random Forest")

result_df = X_test.copy()
result_df["mag"] = mag_test
result_df["absmag_true"] = y_test
result_df["dist_true"] = dist_test
result_df["absmag_pred_rf"] = absmag_pred_rf

result_df.to_csv("ml_test_results.csv", index=False)
print("\n Saved to: ml_test_results.csv")