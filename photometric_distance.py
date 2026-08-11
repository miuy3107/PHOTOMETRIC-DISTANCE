import pandas as pd
import numpy as np

df = pd.read_csv("hygdata_clean.csv")

#distance calculation using mag and absmag 
#note: the absmag in this data is calculated using the parallax dist
#so the result will not have a noticeable margin of error
distance_modulus = df["mag"] - df["absmag"]
df["dist_predicted"] = np.power(10, (distance_modulus + 5)/5)
print(df[["proper", "mag", "absmag", "dist", "dist_predicted"]].head(10))

#computing the margin of error
df["error_abs"] = np.abs(df["dist_predicted"] - df["dist"])
df["error_percent"] = (df["error_abs"] / df["dist"]) * 100

print(f"Margin of error average: {df['error_percent'].mean():.4f} %")
print(f"Margin of error median: {df['error_percent'].median():.4f} %")
print(f"Biggest margin of error: {df['error_percent'].max():.4f} %")

df = df[df["error_percent"] < 0.1]

#listing results
print("\n===== 5 most correct results =====")
print(df.sort_values("error_percent", ascending=True)
        [["proper", "mag", "absmag", "dist", "dist_predicted", "error_percent"]]
        .head(5))
 
print("\n=====  5 least correct results  =====")
print(df.sort_values("error_percent", ascending=False)
        [["proper", "mag", "absmag", "dist", "dist_predicted", "error_percent"]]
        .head(5))


df.to_csv("hygdata_with_predictions.csv", index=False)