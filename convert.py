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
        return ticker  # already looks like Yahoo format
    
    prefix, code = ticker.split(":")
    suffix = mapping.get(prefix, "")
    
    # Special handling for Hong Kong tickers: pad to 4 digits if numeric
    if prefix == "HKG" and code.isdigit():
        code = code.zfill(4)
    
    return code + suffix

# === Main Script ===
input_file = "/Users/ophelianeo/Downloads/Screening here.xlsx"
output_file = "/Users/ophelianeo/Downloads/Screening_converted.xlsx"

# Read Excel column E (index=4)
df = pd.read_excel(input_file, sheet_name=0)
tickers_raw = df.iloc[:, 4]

# Convert
tickers_converted = tickers_raw.apply(convert_ticker)

# Save to new Excel
result_df = pd.DataFrame({
    "Original_Ticker": tickers_raw,
    "Yahoo_Ticker": tickers_converted
})
result_df.to_excel(output_file, index=False)

print(f"✅ Converted {result_df.shape[0]} tickers and saved to {output_file}")
print(result_df.head(20))
