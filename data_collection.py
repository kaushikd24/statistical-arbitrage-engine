"""
Note: This script fetches historical stock data using yfinance.
Yahoo Finance is rate-limited and unstable at times. You may experience
YFRateLimitError or empty responses. Recommended to:
- Download in smaller batches
- Add delays (time.sleep)
- Use cached files when possible
"""

import pandas as pd
import yfinance as yf
import os
from datetime import datetime
import time

#defining stocks tickers

# Define top 3 stocks from 17 sectors

tickers = [
    'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS',
    'ULTRACEMCO.NS', 'AMBUJACEM.NS', 'GRASIM.NS',
    'TCS.NS', 'INFY.NS', 'WIPRO.NS',
    'LT.NS', 'RELINFRA.NS', 'LODHA.NS',
    'HAVELLS.NS', 'BAJAJELEC.NS', 'WHIRLPOOL.NS',
    'HAL.NS', 'BEL.NS', 'BDL.NS',
    'SIEMENS.NS', 'ABB.NS', 'CGPOWER.NS',
    'LICHSGFIN.NS', 'BAJFINANCE.NS',
    'APOLLOHOSP.NS', 'FORTIS.NS', 'NH.NS',
    'HDFCLIFE.NS', 'SBILIFE.NS', 'ICICIPRULI.NS',
    'COALINDIA.NS', 'HINDZINC.NS', 'VEDL.NS',
    'GAIL.NS', 'IGL.NS', 'MGL.NS',
    'NTPC.NS', 'TATAPOWER.NS', 'POWERGRID.NS',
    'RELIANCE.NS', 'IOC.NS', 'BPCL.NS',
    'JSWSTEEL.NS', 'TATASTEEL.NS', 'JINDALSTEL.NS',
    'MRF.NS', 'APOLLOTYRE.NS', 'CEATLTD.NS'
]


# Create ticker dataframe


ticker_df = pd.DataFrame({
    'Ticker': tickers,
    'Start Date': '2018-01-01',
    'End Date': datetime.today().strftime('%Y-%m-%d'),
    'Frequency': '1d'
})

os.makedirs('data', exist_ok=True)

for index, row in ticker_df.iterrows():
    ticker = row['Ticker']
    start = row['Start Date']
    end = row['End Date']
    interval = row['Frequency']
    
    print(f"Downloading data for {ticker} from {start} to {end}...")
    
    try:
        data = yf.download(ticker, start = start, end = end, interval = interval)
        if not data.empty:
            data.to_csv(f'data/{ticker}.csv')
            print(f"Successfully saved {ticker} to data/{ticker}.csv")
        else:
            print(f"No data found for {ticker}")
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        
        

        
        

    
    









    
    
    
    
    