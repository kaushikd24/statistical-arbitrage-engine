import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os

signal_df = pd.read_csv('data/signals.csv', index_col=0, parse_dates=True)

cleaned_df = signal_df.copy()
cleaned_df[:] = "NONE"


dates_index = signal_df.index

# Create cleaned_df with same shape, filled with "NONE"
cleaned_df = pd.DataFrame("NONE", index=signal_df.index, columns=signal_df.columns)

# Loop through each pair 
for pair in signal_df.columns:
    state = "NONE"  # Initial state for the pair

    for date in dates_index:
        signal = signal_df.loc[date, pair]

        if state == "NONE":
            if signal in ["LONG", "SHORT"]:
                cleaned_df.loc[date, pair] = signal
                state = signal
            else:
                cleaned_df.loc[date, pair] = "NONE"

        elif state == "LONG":
            if signal in ["HOLD", "EXIT"]:
                cleaned_df.loc[date, pair] = signal
                if signal == "EXIT":
                    state = "NONE"
            else:
                cleaned_df.loc[date, pair] = "NONE"

        elif state == "SHORT":
            if signal in ["HOLD", "EXIT"]:
                cleaned_df.loc[date, pair] = signal
                if signal == "EXIT":
                    state = "NONE"
            else:
                cleaned_df.loc[date, pair] = "NONE"
                
#save the cleaned_df a as csv in the data folder
cleaned_df.to_csv("data/signals_clean.csv")
