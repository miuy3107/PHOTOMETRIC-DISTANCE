import pandas as pd
df_raw = pd.read_csv("hygdata_clean.csv")  
print(df_raw.iloc[[107846, 107847, 107848]].to_string())