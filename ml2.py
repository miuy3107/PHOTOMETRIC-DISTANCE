"""
ADD LUMINOSITY CLASS FEATURE + REMOVE VARIABLE STARS
=====================================================
Two improvements to the photometric-distance model:

1. REMOVE VARIABLE STARS
   Photometric-distance methods assume that a star's brightness is stable.
   Variable stars change brightness over time (for example, because they
   pulsate or are eclipsing binaries), so a magnitude measured at one moment
   may not represent their standard brightness. They add noise rather than a
   reliable pattern for the model to learn, so they are removed from the data.

2. ADD LUMINOSITY CLASS AS A SECOND FEATURE (alongside ci)
   In a spectral type such as "G2V" or "K1III", the final letter or Roman
   numeral is the luminosity class. It identifies the star's category:
     V   = main-sequence star / dwarf
     III = giant
     I   = supergiant
     D at the beginning (for example, "DA2") = white dwarf
   This helps the model distinguish stars with similar color but very
   different brightness, such as giants and white dwarfs.

Run: python ml2.py
"""

import re  # Regular-expression library for matching text patterns.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# -----------------------------------------------------------------
# 1. LOAD DATA (the newer version includes var, var_min, var_max, and spect)
# -----------------------------------------------------------------
df = pd.read_csv("hygdata_clean.csv")
print(f"Initial number of stars: {len(df)}")


# -----------------------------------------------------------------
# 2. REMOVE VARIABLE STARS
# -----------------------------------------------------------------
# A star is considered variable when either:
#   (a) the "var" column is populated, meaning it is a confirmed variable star
#       with its own identifier (for example, "V636 Cas"), OR
#   (b) the difference between var_max and var_min is greater than 0.1 magnitude,
#       indicating a significant measured brightness variation.
#
# .notna() returns True/False for each row: True when the cell has a value.
# .fillna(0) replaces NaN with 0 before subtraction, avoiding errors.
var_range = df["var_max"].fillna(0) - df["var_min"].fillna(0)
is_variable = df["var"].notna() | (var_range > 0.1)

print(f"Confirmed variable stars to remove: {is_variable.sum()}")

# ~ means NOT: keep only rows that are not variable stars.
df = df[~is_variable].copy()
print(f"Stars remaining after removing variable stars: {len(df)}")


# -----------------------------------------------------------------
# 3. EXTRACT LUMINOSITY CLASS FROM THE "spect" COLUMN
# -----------------------------------------------------------------
def extract_luminosity_class(spect):
    """Return a simplified luminosity class from a spectral-type string.

    Examples include "K1III", "G2V", and "DA2".

    - An empty value (NaN) returns "Unknown".
    - A type starting with "D" is a white dwarf and returns "D".
    - Otherwise, a regular expression finds a Roman-numeral class such as
      I, II, III, IV, V, VI, or VII.

    The ordering in the pattern is important: longer values must appear first
    so that "III" is not incorrectly matched as "I".
    """
    if pd.isna(spect):
        return "Unknown"

    spect = str(spect).strip()

    if spect.startswith("D"):
        return "D"  # White dwarf.

    # re.search() finds the first substring matching the pattern. The | means
    # "or", and alternatives are tried from left to right.
    pattern = r"VII|VI|IV|III|II|Ia|Ib|I|V"
    match = re.search(pattern, spect)

    if match:
        return match.group()
    return "Unknown"


# .apply() runs the function on every value in the "spect" column.
df["lum_class"] = df["spect"].apply(extract_luminosity_class)

print("\n===== Luminosity-class distribution in the data =====")
# .value_counts() counts occurrences of each distinct value.
print(df["lum_class"].value_counts())


# -----------------------------------------------------------------
# 4. ONE-HOT ENCODE lum_class
# -----------------------------------------------------------------
# Machine-learning models do not understand labels such as "III" or "V";
# they use numeric inputs. One-hot encoding converts one text column into
# several 0/1 columns, one for every possible luminosity class. For example,
# lum_class="III" gives lum_class_III = 1 and all other class columns = 0.
lum_dummies = pd.get_dummies(df["lum_class"], prefix="lum_class")

print(f"\nGenerated one-hot columns: {lum_dummies.columns.tolist()}")

# Combine the original ci column and the new one-hot columns into the complete
# feature table (X).
X = pd.concat([df[["ci"]], lum_dummies], axis=1)
y = df["absmag"]
mag_all = df["mag"]
dist_true_all = df["dist"]

print(f"\nNumber of training features: {X.shape[1]} -> {X.columns.tolist()}")


# -----------------------------------------------------------------
# 5. SPLIT TRAIN/TEST DATA (same approach as step 3)
# -----------------------------------------------------------------
X_train, X_test, y_train, y_test, mag_train, mag_test, dist_train, dist_test = \
    train_test_split(X, y, mag_all, dist_true_all, test_size=0.2, random_state=42)


# -----------------------------------------------------------------
# 6. TRAIN A RANDOM FOREST WITH THE NEW FEATURES
# -----------------------------------------------------------------
model_rf_v2 = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf_v2.fit(X_train, y_train)
absmag_pred_v2 = model_rf_v2.predict(X_test)


# -----------------------------------------------------------------
# 7. EVALUATE (a compact version of the step-3 evaluate_distance function)
# -----------------------------------------------------------------
dist_predicted_v2 = np.power(10, (mag_test - absmag_pred_v2 + 5) / 5)
error_percent_v2 = np.abs(dist_predicted_v2 - dist_test) / dist_test * 100
mae_v2 = mean_absolute_error(y_test, absmag_pred_v2)

print("\n===== RESULTS: Random Forest v2 (ci + lum_class; variable stars removed) =====")
print(f"MAE for absmag:                   {mae_v2:.3f}")
print(f"Distance error % - median:        {error_percent_v2.median():.2f} %")
print(f"Distance error % - mean:          {error_percent_v2.mean():.2f} %")

print("\n(Comparison with step 3: Random Forest using only ci, including variable stars:")
print(" MAE=1.236 | median=35.94% | mean=165.21%)")


# -----------------------------------------------------------------
# 8. VIEW FEATURE IMPORTANCE
# -----------------------------------------------------------------
# Random Forest can estimate which features were most useful in making decisions.
importance = pd.Series(model_rf_v2.feature_importances_, index=X.columns)
print("\n===== Feature importance =====")
print(importance.sort_values(ascending=False))


# -----------------------------------------------------------------
# 9. SAVE RESULTS
# -----------------------------------------------------------------
result_df = X_test.copy()
result_df["mag"] = mag_test
result_df["absmag_true"] = y_test
result_df["dist_true"] = dist_test
result_df["absmag_pred_rf_v2"] = absmag_pred_v2
result_df.to_csv("ml_test_results_v2.csv", index=False)
print("\nSaved results to: ml_test_results_v2.csv")
