import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

#okay, so we have imported all the necessary libraries,
#now we can start with the pair selection process
#importing cointegration test from statsmodels
from statsmodels.tsa.stattools import coint

#now, we need to load the data for the pair selection process
#we generated data from data_collection.py file
#the data is stored in the data folder

#loading the data
import os
os.listdir('data')


dir = os.listdir('data')

#create a list of stock names
stock_names = []

for i in dir:
    if i.endswith(".csv") and "final_pairs" not in i:
        stock_names.append(i.replace(".csv", ""))

    
#create a dictionary to store the data for each stock
stock_data = {}
for i in stock_names:
    stock_data[i]=pd.read_csv(f'data/{i}.csv')
    
#create an empty list to store DataFrames for each stock
all_stocks=[]

#Loop through each stock
for ticker, df in stock_data.items():
    #get a copy of the DataFrame with just the 'Close' column
    temp = df.copy()
    
    #reset index to access rows properly
    temp = temp.reset_index()
    
    #keep only rows that have valid dates (skip header rows)
    temp = temp[temp['index']>1]
    
    #select only the price=date and close cols
    temp = temp [['Price', 'Close']]
    
    #add ticker column
    temp['Ticker'] = ticker
    
    all_stocks.append(temp)
    
#Combine all individual DataFramers
cleaned_data = pd.concat(all_stocks, ignore_index=True)

#rename columns
cleaned_data.columns = ['Date', 'Close_Price', 'Ticker']

#preview
combined_df =cleaned_data.pivot(index="Date", columns='Ticker', values="Close_Price")
combined_df.index = pd.to_datetime(combined_df.index)
combined_df.sort_index(inplace=True)

#drop null val
combined_df.dropna(inplace=True)

#normalise the data
combined_df = combined_df.apply(pd.to_numeric, errors='coerce')

normalised_df = combined_df
normalised_df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
normalised_df.dropna(inplace=True)

from statsmodels.tsa.stattools import coint

#loop over all the stocks and store the p-values in a dict
p_value_dict={}
for i in stock_names:
    for y in stock_names:
        score, pvalue, _ = coint(normalised_df[i], normalised_df[y])
        p_value_dict[f"{i} and {y}"] = pvalue

#convert the dict to a DataFrame
pval_df = pd.DataFrame(list(p_value_dict.items()), columns=["Pair", "p-value"])

#sort it
pval_df = pval_df.sort_values(by="p-value").reset_index(drop=True)

pval_df[["Stock A", "Stock B"]] = pval_df["Pair"].str.split(" and ", expand=True)
pval_df = pval_df[["Stock A", "Stock B", "p-value"]]


pval_df = pval_df[pval_df["Stock A"] != pval_df["Stock B"]]

pval_df = pval_df[["Stock A", "Stock B", "p-value"]].sort_values(by="p-value").reset_index(drop=True)

final_df = pval_df[pval_df["p-value"] < 0.05]

final_df.to_csv("data/final_pairs.csv", index=False)







    
    

    

