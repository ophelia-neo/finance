import yfinance as yf

t = yf.Ticker("TSM")   # example ticker
print(t.balance_sheet.index)
print(t.financials.index)
print(t.cashflow.index)
