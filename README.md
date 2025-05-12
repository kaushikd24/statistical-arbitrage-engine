
**Pairs Trading using Statistical Arbitrage**

This repository contains the implementation of a modular and production ready strategy for exploiting **statistical arbitrage** amongst two stocks with high cointegration, which we would call pairs. This system **automates** the entire workflow : from collecting data from APIs to signal generation, with extensibility for machine learning based filtering and robust backtesting. 

Our strategy achieves a median Compounded Annual Growth Rate (CAGR) of around 30%. We have used 47 Indian Equities with data ranging from 1/1/2018 to 31/03/2025.

**{{This is not financial advice, the user of this strategy is encouraged to explore the markets themselves before considering deployment, as stable past returns do not guarentee stable future returns.}}**


The techniques used in this pipeline are Quantitative Analysis, Machine Learning and Risk Management. The pipeline broadly consists of the following steps:
1. Data Collection
2. Pair Selection
3. Spread and Z-score Calculation
4. Signal Generation
5. Backtesting
6. Machine Learning to filter trades
7. Risk Management
8. Backtesting again -- but now with taking inputs from our Machine Learning Model and Risk Management Class.
9. Miscelleanous steps -- performance optimization and trades logging.

**Overview of the Pipeline:**

**Step-1: Data Collection**
Source: Yahoo Finance
1. We collected data of 47 Indian Equities from Yahoo Finance, for this strategy we used daily data (frequency = 1 day). We started with OHLC (Open-High-Low-Close) data, but we used "Close" as an input for our algorithm.
2. Output: we created combined_df.csv as a combined dataframe-csv file for all 47 stock data.
3. We then dropped LODHA.NS from the data as LODHA's data began from 04/2021 and was corrupting the data.

**Stay Tuned for more !**

