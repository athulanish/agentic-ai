import sqlite3
import pandas as pd

df = pd.read_csv("sales_data.csv")
conn = sqlite3.connect("sales.db")
df.to_sql("sales", conn, if_exists="replace", index=False)
conn.close()
print("Loaded", len(df), "rows into sales.db")

