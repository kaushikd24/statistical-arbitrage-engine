import pandas as pd
import numpy as np
import statsmodels.api as sm

combined_df = pd.read_csv("data/combined_df.csv", index_col=0, parse_dates=True)
final_df = pd.read_csv("data/final_pairs.csv", index_col=0, parse_dates=True)

#looping through final_df and combined_df to calculate the spread for each pair

#creating a spread_dict to store the spread of each pair
spread_dict={}

for x,y,_, in final_df.itertuples():
    
    X = combined_df[x]
    Y = combined_df[y]
    
    X = np.array(X)
    Y = np.array(Y)
    
    model = sm.OLS(Y, sm.add_constant(X))
    results = model.fit()
    beta = results.params[1]
    
    spread = Y - beta*X
    
    spread = pd.Series(spread, index=combined_df.index)
    
    spread.name = f"{combined_df[x].name} : {combined_df[y].name}"
    
    spread_dict[f"{combined_df[x].name} : {combined_df[y].name}"] = spread
    

#converting the spread_dict to a dataframe
spread_df = pd.concat(spread_dict.values(), axis=1)

#saving the spread_df to a csv file
spread_df.to_csv("data/spread.csv")




    
