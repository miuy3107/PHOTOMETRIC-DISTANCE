import pandas as pd

#automatically extract ".gz"
df = pd.read_csv("hyg_v44.csv.gz")  
"""print(df.head()) #print the first 5 datas
#print all existed columns in the data
print(df.columns.tolist())"""
#print the sum of stars
#print(f"\n The sum of stars in the data: {len(df)}")

#keep the essential columns
columns_needed = [
    "proper",   #name of the stars
    "mag",      #apparant magnitude (m)
    "absmag",   #absolute magnitude (M)
    "ci",       #color index (B-V)
    "spect",    #spectral type 
    "dist",     #distance calculated by parallax (parsec)
    "var",      #variable star
    "var_min",  
    "var_max",
]

df_selected = df[columns_needed].copy()

df_clean = df_selected.dropna(subset = ["mag", "absmag", "dist", "ci"])
#print(f"\n After dropping mag/absmag/dist/ci: {len(df_clean)} stars left")

#dist = 100000 means that the distance is not recorded 
df_clean = df_clean[df_clean["dist"] < 100000]
df_clean = df_clean[df_clean["dist"] > 0]
#print(f"\n {len(df_clean)} stars left")

df_clean = df_clean.reset_index(drop = True)
print(df_clean.head())
print(df_clean.describe()) #list abnormal datas

df_clean.to_csv("hygdata_clean.csv", index=False)