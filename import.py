import yfinance as yf

# Fetch stock information
stock = yf.Ticker("SBIN.NS")
data1 = stock.info

# Pretty print the dictionary
print("Stock Information:\n")
for key, value in data1.items():
    print(f"{key:30}: {value}")
