"""""      
THIS FILE IS USED TO GENERATE THE FEATURES FOR THE ML GATEKEEPER. THE FOLLOWING FEATURES HAVE BEEN GENERATED:
1. DAYS_HELD: The number of calendar days the trade was held for
2. LABEL: Target variable: 1 if the trade was profitable, 0 otherwise
3. NUM_DIR: Direction of the trade: 1= LONG, 0=SHORT
4. ENTRY_A_REL: Normalised entry price for stock a
5. ENTRY_B_REL: Normalised entry price for stock b
6. ABS_Z: Absolute z-score at the time of signal entry
7. PAIR_ENCODED: Encoded identifier of the pairs
8. PREV_PNL: PnL Label (+1/-1) of the previous trade
9. PAIR_AVG_HOLD: Average holding period of the pair
10. ABS_Z_X_ENTRY_A: Product of absolute z-score and entry price of stock a
11. Z_PER_DAYS_GAP: Z-score divided by the number of days since the last trade
12. PAIR_DIR_COMBO: Product of the number of directions and the pair encoded
13. ROLLING_WINRATE_5: 5 day rolling winrate of the pair
14. ABS_Z_X_ENTRY_A: Product of absolute z-score and entry price of stock a

"""""

import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

trade_log_df = pd.read_csv("data/trade_log.csv")

def label(pnl):
    if pnl > 0:
        label = 1 
    else:
        label = 0
    
    return label

trade_log_df["label"] = trade_log_df["pnl"].apply(label)

def num_dir(dir):
    if dir == 'LONG':
        return 1
    elif dir == 'SHORT':
        return 0

trade_log_df['num_dir'] = trade_log_df['dir'].apply(num_dir)
trade_log_df.drop(columns=['dir'], inplace=True)

def entry_a_rel(entry_a, entry_b):
    entry_a_rel  = entry_a / (entry_a + entry_b)
    return entry_a_rel

def entry_b_rel(entry_a, entry_b):
    entry_b_rel  = entry_b / (entry_a + entry_b)
    return entry_b_rel

trade_log_df['entry_a_rel'] = entry_a_rel(trade_log_df['entry_a'], trade_log_df['entry_b'])
trade_log_df['entry_b_rel'] = entry_b_rel(trade_log_df['entry_a'], trade_log_df['entry_b'])

#bringing back our z-score calculation
spread_df = pd.read_csv("data/spread.csv", index_col=0, parse_dates=True)
pairs=[]
for x in spread_df.columns:
    pairs.append(x)
    
#calcuting the staticstics for each pair

rolling_mean={}
rolling_std={}
z_scores={}

#choosing a window size of 20 days

for x in pairs:
    rolling_mean[x] = spread_df[x].rolling(window=20).mean()
    rolling_std[x] = spread_df[x].rolling(window=20).std()
    z_scores[x] = (spread_df[x] - rolling_mean[x])/rolling_std[x]
    

#creating the dataframe for z scores
z_scores = pd.concat(z_scores, axis=1)

#droping the NaN values (first 19 days would be NaN due to the window size)
z_scores.dropna(inplace=True)

#merging the z-scores with the trade log
# Make sure index is datetime for both
z_scores.index = pd.to_datetime(z_scores.index)
trade_log_df["entry_dt"] = pd.to_datetime(trade_log_df["entry_dt"])

# Applying lookup row by row
trade_log_df["z_score"] = trade_log_df.apply(
    lambda row: z_scores.at[row["entry_dt"], row["pair"]], axis=1
)

# Final step: absolute z-score
trade_log_df["abs_z"] = trade_log_df["z_score"].abs()

#dropping the columns we don't need
trade_log_df.drop(columns=['entry_dt', 'exit_dt','entry_a', 'entry_b', 'exit_a', 'exit_b', 'pnl','z_score'], inplace=True)


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
trade_log_df["pair_encoded"] = le.fit_transform(trade_log_df["pair"])
trade_log_df.drop(columns=["pair"], inplace=True)

#this calculates the PnL from the last trade on the same pair
trade_log_df['prev_pnl'] = (
    trade_log_df.groupby('pair_encoded')["label"].shift(1).map({1:1, 0:-1}).fillna(0)
)

#this calculates the 5 day rolling winrate for each pair
trade_log_df["rolling_winrate_5"] =(
    trade_log_df.groupby('pair_encoded')["label"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
)

#bringing back the original entry_dt
original = pd.read_csv("data/trade_log.csv", parse_dates=["entry_dt"])
trade_log_df["entry_dt"] = original["entry_dt"]

#this calculates the number of days since the last trade on the same pair   
last_trade_dt = (
    trade_log_df.groupby("pair_encoded")["entry_dt"]
    .shift(1)
)

trade_log_df["days_since_last"] = (trade_log_df["entry_dt"] - last_trade_dt).dt.days.fillna(999)
trade_log_df.drop(columns=['entry_dt'], inplace=True)

#this calculates the average holding period for each pair
trade_log_df["pair_avg_hold"] = (
    trade_log_df.groupby("pair_encoded")["days_held"]
    .expanding()
    .mean()
    .reset_index(level=0, drop=True)
)

#this calculates the product of the absolute z-score and the entry price of stock a
trade_log_df["abs_z_x_entry_a"] = trade_log_df["abs_z"] * trade_log_df["entry_a_rel"]

#this calculates the z-score divided by the number of days since the last trade
trade_log_df["z_per_days_gap"]  = trade_log_df["abs_z"] / (trade_log_df["days_since_last"] + 1)

#this calculates the product of the number of directions and the pair encoded
trade_log_df["pair_dir_combo"]  = trade_log_df["num_dir"] * trade_log_df["pair_encoded"]

#filling the NaN values with 0
trade_log_df.fillna(0, inplace=True)

#saving the dataframe as a csv file
trade_log_df.to_csv("data/trade_ml_fodder.csv", index=False)




