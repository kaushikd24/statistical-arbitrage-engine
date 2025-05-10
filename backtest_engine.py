import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

prices = pd.read_csv('data/combined_df.csv')
signals = pd.read_csv('data/signals_clean.csv')

prices.rename(columns={"Price": "Date"}, inplace=True)
signals.rename(columns={"Price": "Date"}, inplace=True)

prices["Date"] = pd.to_datetime(prices["Date"])
signals["Date"] = pd.to_datetime(signals["Date"])

prices.set_index("Date", inplace=True)
signals.set_index("Date", inplace=True)

#include only common dates (although this is already the case, but double checking)
dates = prices.index.intersection(signals.index)
prices = prices.loc[dates]
signals = signals.loc[dates]

#creating the backtesting engine

#creating the trade log, a list of dictionaries
trade_log =[]

#creating the daily PnL, a list of dictionaries again
equity = []

#creating the key which is thepair column name
open_trades = {}

#defining symbols, which means the stock pairs per leg
def leg_symbols(pair_col):
    a, b = [x.strip() for x in pair_col.split(":")]
    return a, b

for dt in dates:
    daily_pnl = 0.0
    
    for pair_col in signals.columns:
        sig = signals.at[dt, pair_col]
        if sig == "NONE" or pd.isna(sig):
            continue
        
        ticker_a, ticker_b = leg_symbols(pair_col)
        px_a = prices.at[dt, ticker_a]
        px_b = prices.at[dt, ticker_b]
        
        #recording for long and short
        if sig in ("LONG", "SHORT") and pair_col not in open_trades:
            direction = +1 if sig  == "LONG" else -1
            open_trades[pair_col] = {
                "entry_date": dt,
                "dir" : direction,
                "px_a0": px_a,
                "px_b0": px_b,
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
            }
            
            #while exiting
        elif sig == "EXIT" and pair_col in open_trades:
            tr = open_trades.pop(pair_col)
            direction = tr["dir"]
            pnl = direction*(px_a - tr["px_a0"]) - direction*(px_b - tr["px_b0"])
            
            trade_log.append({
                "pair": pair_col,
                "entry_dt": tr["entry_date"],
                "exit_dt": dt,
                "dir": "LONG" if direction ==1 else "SHORT",
                "entry_a" : tr["px_a0"],
                "entry_b" : tr["px_b0"],
                "exit_a" : px_a,
                "pnl" : pnl,
                "days_held" : (dt - tr["entry_date"]).days,
            })
            
            daily_pnl += pnl
            
    #mark to market unrealised PnL for open trades
    for tr in open_trades.values():
        direction = tr["dir"]
        cur_px_a = prices.at[dt, tr["ticker_a"]]
        cur_px_b = prices.at[dt, tr["ticker_b"]]
        daily_pnl = direction*(cur_px_a - tr["px_a0"]) - direction*(cur_px_b - tr["px_b0"])
    
    equity.append({"Date": dt, "equity": daily_pnl})
    
#force closing all open trades at the last day of the data

if open_trades:
    last_dt = dates[-1]
    for pair_col, tr in open_trades.items():
        px_a = prices.at[last_dt, tr["ticker_a"]]
        px_b = prices.at[last_dt, tr["ticker_b"]]
        direction = tr["dir"]
        pnl = direction*(px_a - tr["px_a0"]) - direction*(px_b - tr["px_b0"])
        
        
        trade_log.append({
            "pair": pair_col,
            "entry_dt": tr["entry_date"],
            "exit_dt": last_dt,
            "dir": "LONG" if direction ==1 else "SHORT",
            "entry_a" : tr["px_a0"],
            "entry_b" : tr["px_b0"],
            "exit_a" : px_a,
            "exit_b" : px_b,
            "pnl" : pnl,
            "days_held" : (last_dt - tr["entry_date"]).days, 
        })
        
        equity[-1]["equity"] += pnl
        
#converting the lists into dataframes

trade_log_df = pd.DataFrame(trade_log)
equity_df = pd.DataFrame(equity).set_index("Date").cumsum()

#converting the trade and equity dataframes into csv files
trade_log_df.to_csv("data/trade_log.csv", index=False)
equity_df.to_csv("data/equity.csv", index=True)

print("Finished backtesting. # of trades: ", len(trade_log_df))


