import yfinance as yf
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def calculate_ratios(ticker):
    """Pull financials and ratios for a single ticker"""
    results = {}
    stock = yf.Ticker(ticker)

    bs = stock.balance_sheet
    is_ = stock.financials
    cf = stock.cashflow
    info = stock.info

    if bs.empty or is_.empty or cf.empty:
        print(f"⚠️ No financial data found for {ticker}")
        return pd.DataFrame()

    cols = bs.columns
    latest, prev = cols[0], (cols[1] if len(cols) > 1 else None)

    def get(df, items, col=latest):
        for item in items:
            if item in df.index and col in df.columns:
                val = df.loc[item, col]
                if pd.notna(val):
                    return val
        return None

    def get_avg(df, items):
        cur = get(df, items, latest)
        prev_val = get(df, items, prev) if prev else None
        return (cur + prev_val) / 2 if cur and prev_val else cur

    # === Balance Sheet ===
    current_assets = get(bs, ["Current Assets"])
    current_liabilities = get(bs, ["Current Liabilities"])
    cash = get(bs, ["Cash And Cash Equivalents"])
    inventory_latest = get(bs, ["Inventory"])
    total_assets_latest = get(bs, ["Total Assets"])
    total_equity_latest = get(bs, ["Total Equity Gross Minority Interest"])

    short_debt = get(bs, ["Current Debt And Capital Lease Obligation"]) or 0
    long_debt = get(bs, ["Long Term Debt And Capital Lease Obligation"]) or 0
    total_debt = get(bs, ["Total Debt"]) or (short_debt + long_debt)

    avg_inventory = get_avg(bs, ["Inventory"])
    avg_total_assets = get_avg(bs, ["Total Assets"])
    avg_total_equity = get_avg(bs, ["Total Equity Gross Minority Interest"])

    # === Income Statement ===
    revenue = get(is_, ["Total Revenue"])
    gross_profit = get(is_, ["Gross Profit"])
    net_income = get(is_, ["Net Income"])
    cogs = get(is_, ["Cost Of Revenue"])
    interest_expense = get(is_, ["Interest Expense"])
    income_tax = get(is_, ["Tax Provision"])
    operating_income = get(is_, ["Operating Income"])

    # === Cash Flow ===
    operating_cf = get(cf, ["Operating Cash Flow"])
    free_cf = get(cf, ["Free Cash Flow"])  # optional extra

    # === EBIT ===
    ebit = get(is_, ["EBIT"])
    if ebit is None and net_income is not None:
        ebit = net_income + (interest_expense or 0) + (income_tax or 0)

    ratios = {}

    # Liquidity
    if current_assets and current_liabilities:
        ratios["Current_Ratio"] = current_assets / current_liabilities
        ratios["Quick_Ratio"] = (current_assets - (inventory_latest or 0)) / current_liabilities
        ratios["Cash_Ratio"] = (cash or 0) / current_liabilities

    # Leverage
    if total_debt and total_equity_latest:
        ratios["Debt_to_Equity"] = total_debt / total_equity_latest
    if ebit and interest_expense:
        ratios["Interest_Coverage"] = ebit / interest_expense
    if total_assets_latest and total_debt:
        ratios["Debt_Ratio"] = total_debt / total_assets_latest

    # Profitability
    if revenue and gross_profit:
        ratios["Gross_Profit_Margin"] = gross_profit / revenue
    if revenue and net_income:
        ratios["Net_Profit_Margin"] = net_income / revenue
    if net_income and avg_total_assets:
        ratios["ROA"] = net_income / avg_total_assets
    if net_income and avg_total_equity:
        ratios["ROE"] = net_income / avg_total_equity
    if total_assets_latest and ebit and current_liabilities is not None:
        cap_emp = total_assets_latest - (current_liabilities or 0)
        if cap_emp != 0:
            ratios["ROCE"] = ebit / cap_emp

    # Efficiency
    if cogs and avg_inventory:
        ratios["Inventory_Turnover"] = cogs / avg_inventory
    if revenue and avg_total_assets:
        ratios["Asset_Turnover"] = revenue / avg_total_assets

    # Cash Flow
    if operating_cf:
        ratios["Operating_CF"] = operating_cf  # 🔥 raw operating cash flow
    if operating_cf and current_liabilities:
        ratios["Operating_CF_Ratio"] = operating_cf / current_liabilities
    if free_cf:
        ratios["Free_CF"] = free_cf  # 🔥 free cash flow too (optional)

    # Operating Ratio
    if revenue and operating_income is not None:
        operating_expenses = revenue - operating_income
        ratios["Operating_Ratio"] = operating_expenses / revenue

    # Valuation & Market Stats
    ratios["Forward_PE"] = info.get("forwardPE")
    ratios["EV_EBITDA"] = (
        info.get("enterpriseValue") / info.get("ebitda")
        if info.get("enterpriseValue") and info.get("ebitda") else None
    )
    ratios["Price_to_Book"] = info.get("priceToBook")
    ratios["Beta"] = info.get("beta")
    ratios["50DMA"] = info.get("fiftyDayAverage")
    ratios["200DMA"] = info.get("twoHundredDayAverage")
    ratios["Market_Cap"] = info.get("marketCap")
    ratios["Industry"] = info.get("industry")

    results[ticker] = ratios
    return pd.DataFrame(results).T.round(4)


def main():
    ticker = "6098.T"   # 🔧 change this to any single ticker
    print("="*80)
    print(f"FINANCIAL RATIOS for {ticker}")
    print("="*80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df_ratios = calculate_ratios(ticker)

    # Save with timestamp
    output_file = '/Users/ophelianeo/Documents/work!/finance/single.xlsx'
    df_ratios.to_excel(output_file, index=True)

    print(f"✅ Saved results to {output_file}")
    print(df_ratios.head())


if __name__ == "__main__":
    main()
