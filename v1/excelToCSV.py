import pandas as pd

# ==============================
# CONFIG
# ==============================
INPUT_EXCEL = "../dataset.xlsx"
OUTPUT_CSV = "demand_data.csv"

# ==============================
# LOAD EXCEL
# ==============================
df = pd.read_excel(INPUT_EXCEL)

print("Original Columns:", df.columns.tolist())

# ==============================
# CASE 1: Already correct format
# ==============================
if set(['date', 'sku_id', 'demand']).issubset(df.columns):
    print("Detected LONG format → no conversion needed")
    df_long = df.copy()

# ==============================
# CASE 2: Wide format
# ==============================
elif 'date' in df.columns or 'Date' in df.columns:

    print("Detected WIDE format → converting to long format")

    if 'Date' in df.columns:
        df.rename(columns={'Date': 'date'}, inplace=True)

    df_long = df.melt(
        id_vars=['date'],
        var_name='sku_id',
        value_name='demand'
    )

# ==============================
# ERROR CASE
# ==============================
else:
    raise ValueError(
        "Unsupported Excel format. Expected either:\n"
        "1. date, sku_id, demand\n"
        "2. date + multiple SKU columns"
    )

# ==============================
# DATE HANDLING
# ==============================
df_long['date'] = pd.to_datetime(df_long['date'], errors='coerce')
df_long = df_long.dropna(subset=['date'])

# ==============================
# AGGREGATE DUPLICATES
# ==============================
df_long['demand'] = df_long['demand'].fillna(0)

df_long = df_long.groupby(['sku_id', 'date'], as_index=False)['demand'].sum()

# ==============================
# SORT
# ==============================
df_long = df_long.sort_values(by=['sku_id', 'date']).reset_index(drop=True)

# ==============================
# 🔥 ENFORCE COLUMN ORDER
# ==============================
df_long = df_long[['date', 'sku_id', 'demand']]

# ==============================
# SAVE
# ==============================
df_long.to_csv(OUTPUT_CSV, index=False, date_format='%d-%m-%Y')

print(f"✅ Converted successfully → {OUTPUT_CSV}")
print(df_long.head())