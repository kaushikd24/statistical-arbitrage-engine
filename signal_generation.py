import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm

spread_df = pd.read_csv("data/spread.csv", index_col=0, parse_dates=True)

#creating a list of cols to help us in calc the z scores

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

#loading signal rules

params = {}

with open("signal_logic.txt", "r") as f:
    for line in f:
        if "=" in line:
            key, value = line.strip().split("=")
            params[key.strip()] = float(value.strip())

#creating a function to generate the signal
ENTRY_THRESHOLD = params['ENTRY_THRESHOLD']
EXIT_THRESHOLD = params['EXIT_THRESHOLD']

def generate_signal(z_score):
    if z_score<-ENTRY_THRESHOLD:
        return "LONG"
    elif z_score>ENTRY_THRESHOLD:
        return "SHORT"
    elif abs(z_score)<EXIT_THRESHOLD:
        return "EXIT"
    else:
        return "HOLD"
    
#creating a dict to store signals for each pair
signals = {}

for x in z_scores.columns:
    signals[x] = z_scores[x].apply(generate_signal)
    

#converting the dict into a dataframe
signal = pd.DataFrame(signals, index=z_scores.index)

#creating a csv file for the signals
signal.to_csv('data/signals.csv')

#creating a csv file for the z scores
z_scores.to_csv('data/z_scores.csv')

