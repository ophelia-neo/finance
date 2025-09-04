import pandas as pd

# File paths
ratios_file = "/Users/ophelianeo/Documents/work!/finance/Financial Ratios.xlsx"
screening_file = "/Users/ophelianeo/Documents/work!/finance/Screening_converted.xlsx"
output_file = "/Users/ophelianeo/Documents/work!/finance/Financial_Ratios_Reordered.xlsx"

# === Load ratios ===
ratios_df = pd.read_excel(ratios_file, sheet_name="Financial_Ratios", index_col=0)

# === Load screening tickers (col 2 = index 1) ===
screening_df = pd.read_excel(screening_file, sheet_name="Sheet1")
desired_order = screening_df.iloc[:, 1].dropna().astype(str).str.strip().tolist()

# === Reorder ratios according to screening ===
reordered = ratios_df.reindex(desired_order)

# Save to new Excel
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    reordered.to_excel(writer, sheet_name="Financial_Ratios_Reordered")

print("✅ Done! Saved reordered ratios to:", output_file)
