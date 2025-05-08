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


