import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

from statsmodels.tsa.stattools import coint

dir = os.listdir("data")

#create a list of stock names, which are the names of the tickers in the data folder
stock_names = [filename.replace(".csv", "") for filename in dir if filename.endswith(".csv") and "final_pairs" not in filename]

#create a dictionary to store stock data, and reference it by stock name
stock_data = {}

for ticker in stock_names:
    try:
        df = pd.read_csv(f"data/{ticker}.csv")  # Fixed path from "date" to "data"
        
        # Reset index to access rows properly
        df = df.reset_index()
        
        # Keep only rows that have valid dates (skip header rows)
        df = df[df['index'] > 1]
        
        # Select only the price=date and close cols
        temp = df[['Price', 'Close']]
        
        temp = temp.dropna()
        
        temp["Price"] = pd.to_datetime(temp["Price"])
        
        temp = temp.set_index("Price").sort_index()
        stock_data[ticker] = temp["Close"].astype(float).rename(ticker)
        
    except Exception as e:
        print(f"Error loading {ticker}: {e}")

#LODHA.NS has data from 2021, so we would remove it from stock_data
if "LODHA.NS" in stock_data:
    del stock_data["LODHA.NS"]
    stock_names.remove("LODHA.NS")
    
#combining all series into a single DataFrame
combined_df = pd.concat(stock_data.values(), axis=1, join="inner")

#set column names using ticker list 
combined_df.columns = stock_data.keys()  # Fixed comma to period

#sort by date 
combined_df = combined_df.sort_index()

results = []

tickers = list(combined_df.columns)

for i in range(len(tickers)):
    for j in range(i+1, len(tickers)):
        stock_a = tickers[i]
        stock_b = tickers[j]
        
        series_a = combined_df[stock_a]
        series_b = combined_df[stock_b]
        
        #run the engle-granger cointegration test
        try:
            _, p_value, _ = coint(series_a, series_b)
            results.append((stock_a, stock_b, p_value))
            
        except Exception as e:
            print(f"Skipped {stock_a} and {stock_b}: {e}")
            

#convert to a DataFrame
pval_df = pd.DataFrame(results, columns=["Stock A", "Stock B", "p-value"])

#filter based on p-value threshold, in our case it is 0.05
final_df = pval_df[pval_df["p-value"] < 0.05]

#sort by p-value
final_df = final_df.sort_values(by="p-value")

#saving the final pairs into a csv file 
final_df.to_csv("data/final_pairs.csv", index=False)

        
    