import yfinance as yf
import pandas as pd
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_tickers(excel_file, column_name=None, column_index=None, sheet_name=0):
    """
    Load tickers from Excel file
    
    Parameters:
    excel_file: str - path to Excel file
    column_name: str - column header name containing tickers
    column_index: int - column index (0=A, 1=B, 2=C, etc.)
    sheet_name: str/int - sheet name or index
    """
    try:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)

        if column_name and column_name in df.columns:
            tickers = df[column_name].dropna().astype(str).str.strip().tolist()
        elif column_index is not None:
            tickers = df.iloc[:, column_index].dropna().astype(str).str.strip().tolist()
        else:
            tickers = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()

        tickers = list(set([t for t in tickers if t and t != 'nan']))

        print(f"✅ Loaded {len(tickers)} unique tickers from {excel_file}")
        return tickers

    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        return []

def calculate_ratios(ticker):
    """Calculate financial ratios for a given ticker"""
    try:
        stock = yf.Ticker(ticker)
        
        # Download financial statements
        bs = stock.balance_sheet
        is_ = stock.financials  
        cf = stock.cashflow
        
        if bs.empty or is_.empty or cf.empty:
            return {}
            
        latest = bs.columns[0]
        
        def get(df, items):
            for item in items:
                if item in df.index:
                    value = df.loc[item, latest]
                    if pd.isna(value):
                        continue
                    return value
            return None
        
        # Balance sheet items
        current_assets = get(bs, ["Total Current Assets", "Current Assets"])
        current_liabilities = get(bs, ["Total Current Liabilities", "Current Liabilities"])
        cash = get(bs, ["Cash And Cash Equivalents", "Cash"])
        inventory = get(bs, ["Inventory"])
        total_assets = get(bs, ["Total Assets"])
        total_debt = get(bs, ["Total Debt", "Long Term Debt", "Short Long Term Debt"])
        total_equity = get(bs, ["Total Stockholder Equity", "Total Equity", "Stockholders Equity"])
        
        # Income statement items
        revenue = get(is_, ["Total Revenue", "Revenue"])
        gross_profit = get(is_, ["Gross Profit"])
        ebit = get(is_, ["EBIT", "Operating Income", "Ebit"])
        net_income = get(is_, ["Net Income"])
        cogs = get(is_, ["Cost Of Revenue", "Cost of Goods Sold", "Total Costs"])
        interest_expense = get(is_, ["Interest Expense"])
        
        # Cash flow items
        operating_cf = get(cf, ["Total Cash From Operating Activities", "Operating Cash Flow"])
        
        ratios = {}
        
        # === Liquidity Ratios ===
        if current_assets and current_liabilities and current_liabilities != 0:
            ratios["Current_Ratio"] = current_assets / current_liabilities
            ratios["Quick_Ratio"] = (current_assets - (inventory or 0)) / current_liabilities
            ratios["Cash_Ratio"] = (cash or 0) / current_liabilities
        
        # === Leverage Ratios ===
        if total_debt and total_equity and total_equity != 0:
            ratios["Debt_to_Equity"] = total_debt / total_equity
            
        if ebit and interest_expense and interest_expense != 0:
            ratios["Interest_Coverage"] = ebit / abs(interest_expense)
            
        if total_assets and total_debt and total_assets != 0:
            ratios["Debt_Ratio"] = total_debt / total_assets
        
        # === Profitability Ratios ===
        if revenue and gross_profit and revenue != 0:
            ratios["Gross_Profit_Margin"] = gross_profit / revenue
            
        if revenue and net_income and revenue != 0:
            ratios["Net_Profit_Margin"] = net_income / revenue
            
        if total_assets and net_income and total_assets != 0:
            ratios["ROA"] = net_income / total_assets
            
        if total_equity and net_income and total_equity != 0:
            ratios["ROE"] = net_income / total_equity
            
        if total_assets and ebit and current_liabilities and (total_assets - current_liabilities) != 0:
            ratios["ROCE"] = ebit / (total_assets - current_liabilities)
        
        # === Efficiency Ratios ===
        if cogs and inventory and inventory != 0:
            ratios["Inventory_Turnover"] = cogs / inventory
            
        if total_assets and revenue and total_assets != 0:
            ratios["Asset_Turnover"] = revenue / total_assets
        
        # === Additional Ratios ===
        if operating_cf and current_liabilities and current_liabilities != 0:
            ratios["Operating_CF_Ratio"] = operating_cf / current_liabilities
            
        return ratios
        
    except Exception:
        return {}

def process_tickers_batch(tickers, batch_size=50, delay=1.0):
    """Process tickers in batches with progress tracking"""
    
    results = {}
    failed_tickers = []
    total = len(tickers)
    
    print(f"\n🚀 Starting analysis of {total} tickers...")
    print(f"📦 Batch size: {batch_size}, Delay: {delay}s between requests")
    print("="*80)
    
    start_time = time.time()
    
    for i in range(0, total, batch_size):
        batch = tickers[i:i + batch_size]
        batch_start = time.time()
        
        print(f"\n📊 Processing batch {i//batch_size + 1}/{(total-1)//batch_size + 1} "
              f"(Tickers {i+1}-{min(i+batch_size, total)})")
        
        for j, ticker in enumerate(batch):
            try:
                ratios = calculate_ratios(ticker)
                if ratios:
                    results[ticker] = ratios
                    status = f"✅ {ticker}: {len(ratios)} ratios"
                else:
                    failed_tickers.append(ticker)
                    status = f"⚠️  {ticker}: No data"
                    
            except Exception:
                failed_tickers.append(ticker)
                status = f"❌ {ticker}: Error"
            
            overall_progress = ((i + j + 1) / total) * 100
            print(f"  [{j+1:2d}/{len(batch):2d}] {status} (Overall: {overall_progress:.1f}%)")
            
            if j < len(batch) - 1:
                time.sleep(delay)
        
        batch_time = time.time() - batch_start
        elapsed_time = time.time() - start_time
        
        success_count = len(results)
        print(f"  📈 Batch completed in {batch_time:.1f}s | "
              f"Success rate: {success_count}/{i+len(batch)} "
              f"({success_count/(i+len(batch))*100:.1f}%) | "
              f"Elapsed: {elapsed_time/60:.1f}min")
    
    return results, failed_tickers

def main():
    """Main function to run the analysis"""
    
    # === SETTINGS ===
    excel_file = "/Users/ophelianeo/Downloads/Screening_converted.xlsx"
    sheet_name = "Sheet1"
    batch_size = 30
    delay = 0.5
    
    print("="*80)
    print("FINANCIAL RATIOS CALCULATOR - BULK ANALYSIS")
    print("="*80)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load tickers from column E
    tickers = load_tickers(excel_file, column_index=1, sheet_name=sheet_name)

    if not tickers:
        print("❌ No tickers loaded. Please check your Excel file.")
        return
    
    print(f"\n📋 Sample tickers: {tickers[:10]}")
    if len(tickers) > 10:
        print(f"    ... and {len(tickers)-10} more")
    
    results, failed_tickers = process_tickers_batch(tickers, batch_size, delay)
    
    if results:
        df = pd.DataFrame(results).T.round(4)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"financial_ratios_{timestamp}.xlsx"
        
        try:
            with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Financial_Ratios")
                df.describe().to_excel(writer, sheet_name="Summary_Statistics")
                if failed_tickers:
                    pd.DataFrame(failed_tickers, columns=["Failed_Ticker"]).to_excel(
                        writer, sheet_name="Failed_Tickers", index=False
                    )
            
            print("\n" + "="*80)
            print("📊 ANALYSIS COMPLETE!")
            print("="*80)
            print(f"✅ Successfully analyzed: {len(results)} tickers")
            print(f"❌ Failed to analyze: {len(failed_tickers)} tickers")
            print(f"📁 Results saved to: {output_file}")
            
            print(f"\n📈 Sample results (first 5 companies):")
            print(df.head())
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            print(df.head())
    else:
        print("\n❌ No data retrieved for any tickers.")
        if failed_tickers:
            print(f"Failed tickers: {failed_tickers[:20]}...")

if __name__ == "__main__":
    main()
