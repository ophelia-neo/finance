import pandas as pd

def convert_ticker(bloomberg_ticker):
    mapping = {
        "TKS": ".T", "TAI": ".TW", "HKG": ".HK", "BOM": ".BO", "BSE": ".BO",
        "KRX": ".KS", "SHG": ".SS", "SHE": ".SZ", "ASX": ".AX", "KLS": ".KL",
        "JKT": ".JK", "BKK": ".BK", "SES": ".SI", "NZE": ".NZ", "JSE": ".JO"
    }
    
    if pd.isna(bloomberg_ticker):
        return None
    
    ticker = str(bloomberg_ticker).strip()
    if ":" not in ticker:
        return ticker  # already Yahoo-style
    
    prefix, code = ticker.split(":")
    suffix = mapping.get(prefix, "")
    
    # Pad HK tickers to 4 digits if numeric
    if prefix == "HKG" and code.isdigit():
        code = code.zfill(4)
    
    return code + suffix


# === Main Script ===
file_path = "/Users/ophelianeo/Documents/work!/finance/Tickers.xlsx"

# Read Excel
df = pd.read_excel(file_path, sheet_name=0)

# Convert tickers in column C (index=2)
converted = df.iloc[:, 2].apply(convert_ticker)

# If column D exists, overwrite it, else insert it
if df.shape[1] > 3:
    df.iloc[:, 3] = converted
    df.rename(columns={df.columns[3]: "Yahoo_Ticker"}, inplace=True)
else:
    df.insert(3, "Yahoo_Ticker", converted)

# Save back
df.to_excel(file_path, index=False)

print(f"✅ Added Yahoo_Ticker to column D in {file_path}")
print(df.head(10))
