import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import joblib
import argparse
import json
from risk_management.risk_manager import RiskManager
from sklearn.ensemble import RandomForestClassifier
from performance_logger import PerformanceLogger

def run_backtest(risk_config, ml_threshold=0.4, leverage=1.0, strategy_name="Default Strategy"):
    """
    Run a backtest with the specified parameters and log the results.
    
    Args:
        risk_config (dict): Risk management parameters
        ml_threshold (float): ML confidence threshold for filtering trades
        leverage (float): Leverage multiplier for position sizing
        strategy_name (str): Name of the strategy configuration
        
    Returns:
        dict: Performance metrics
    """
    print(f"\n===== Running Backtest: {strategy_name} =====")
    print(f"ML Threshold: {ml_threshold}, Leverage: {leverage}x")
    print(f"Risk Config: {json.dumps(risk_config, indent=2)}")
    
    # Load the ML model for trade gatekeeping
    try:
        model = joblib.load("ml_gatekeeper/RFC_model.pkl")
        use_ml = True
        print("ML gatekeeper loaded successfully")
    except:
        use_ml = False
        print("ML gatekeeper not found, continuing without ML filtering")

    # Initialize Risk Manager
    risk_mgr = RiskManager(nav_start=1000000, config=risk_config)

    # Load price and signal data
    prices = pd.read_csv('data/combined_df.csv')
    signals = pd.read_csv('data/signals_clean.csv')
    z_scores = pd.read_csv('data/z_scores.csv')

    # Format data
    prices.rename(columns={"Price": "Date"}, inplace=True)
    signals.rename(columns={"Price": "Date"}, inplace=True)
    z_scores.rename(columns={"Price": "Date"}, inplace=True)

    prices["Date"] = pd.to_datetime(prices["Date"])
    signals["Date"] = pd.to_datetime(signals["Date"])
    z_scores["Date"] = pd.to_datetime(z_scores["Date"])

    prices.set_index("Date", inplace=True)
    signals.set_index("Date", inplace=True)
    z_scores.set_index("Date", inplace=True)

    # Include only common dates
    dates = prices.index.intersection(signals.index)
    prices = prices.loc[dates]
    signals = signals.loc[dates]
    z_scores = z_scores.loc[dates.intersection(z_scores.index)]

    # ML feature prep function (simplified version of what's in feature_eng.py)
    def prepare_ml_features(pair, entry_date, ticker_a, ticker_b, entry_a, entry_b, direction, z_score):
        """Generate features for ML model prediction"""
        # Create a dict with all expected features in the exact order they appear in trade_ml_fodder.csv
        features = {
            'days_held': 0,  # Will be updated during the trade
            'num_dir': 1 if direction == 1 else 0,  # LONG=1, SHORT=0
            'entry_a_rel': entry_a / (entry_a + entry_b),
            'entry_b_rel': entry_b / (entry_a + entry_b),
            'abs_z': abs(z_score),
            'pair_encoded': hash(pair) % 50,  # Simple way to encode pairs
            'prev_pnl': 0,
            'rolling_winrate_5': 0.5,
            'days_since_last': 999,
            'pair_avg_hold': 5,
            'abs_z_x_entry_a': abs(z_score) * (entry_a / (entry_a + entry_b)),
            'z_per_days_gap': abs(z_score) / 999,
            'pair_dir_combo': (1 if direction == 1 else 0) * (hash(pair) % 50)
        }
        
        # Ensure the order matches exactly with what was used for training
        # This is important because scikit-learn models can be sensitive to feature order
        feature_order = [
            'days_held', 'num_dir', 'entry_a_rel', 'entry_b_rel', 'abs_z',
            'pair_encoded', 'prev_pnl', 'rolling_winrate_5', 'days_since_last',
            'pair_avg_hold', 'abs_z_x_entry_a', 'z_per_days_gap', 'pair_dir_combo'
        ]
        
        # Create DataFrame with columns in the correct order
        return pd.DataFrame({col: [features[col]] for col in feature_order})

    # Backtesting setup
    trade_log = []
    equity = []
    open_trades = {}
    rejected_trades = 0
    accepted_trades = 0

    # Helper function to extract ticker symbols from pair column name
    def leg_symbols(pair_col):
        a, b = [x.strip() for x in pair_col.split(":")]
        return a, b

    # Run the backtest
    for dt in dates:
        daily_pnl = 0.0
        
        for pair_col in signals.columns:
            sig = signals.at[dt, pair_col]
            if sig == "NONE" or pd.isna(sig):
                continue
            
            ticker_a, ticker_b = leg_symbols(pair_col)
            try:
                px_a = prices.at[dt, ticker_a]
                px_b = prices.at[dt, ticker_b]
                z_score = z_scores.at[dt, pair_col] if pair_col in z_scores.columns else 0
            except KeyError:
                continue
            
            # Check for new entries
            if sig in ("LONG", "SHORT") and pair_col not in open_trades:
                direction = +1 if sig == "LONG" else -1
                
                # Risk management pre-trade check
                risk_check = risk_mgr.check_pre_trade(
                    pair=pair_col,
                    date=dt,
                    price=px_a,  # Using first stock's price
                    spread_vol=1.0  # Placeholder, can be calculated from z_scores
                )
                
                if not risk_check["allow"]:
                    print(f"Trade rejected by risk manager: {risk_check['reason']}")
                    rejected_trades += 1
                    continue
                    
                # ML gatekeeper check
                if use_ml:
                    ml_features = prepare_ml_features(
                        pair=pair_col,
                        entry_date=dt, 
                        ticker_a=ticker_a,
                        ticker_b=ticker_b,
                        entry_a=px_a,
                        entry_b=px_b,
                        direction=direction,
                        z_score=z_score
                    )
                    
                    # Get probability of success from model
                    success_prob = model.predict_proba(ml_features)[0][1]
                    
                    # Skip trade if ML model gives low probability of success
                    if success_prob < ml_threshold:
                        print(f"Trade rejected by ML: {pair_col}, prob={success_prob:.2f}")
                        rejected_trades += 1
                        continue
                
                # Record the trade
                open_trades[pair_col] = {
                    "entry_date": dt,
                    "dir": direction,
                    "px_a0": px_a,
                    "px_b0": px_b,
                    "ticker_a": ticker_a,
                    "ticker_b": ticker_b,
                    "qty": risk_check["qty"] if risk_check["qty"] > 0 else 200  # Default quantity
                }
                
                # Log the trade with risk manager
                risk_mgr.log_trade(pair_col, len(trade_log))
                accepted_trades += 1
                
            # Check for exits
            elif sig == "EXIT" and pair_col in open_trades:
                tr = open_trades.pop(pair_col)
                direction = tr["dir"]
                
                # Calculate PnL with leverage effect
                pnl = leverage * direction * (px_a - tr["px_a0"]) * tr["qty"] - leverage * direction * (px_b - tr["px_b0"]) * tr["qty"]
                
                # Close position in risk manager
                risk_mgr.close_trade(pair_col)
                
                # Add to trade log
                trade_log.append({
                    "pair": pair_col,
                    "entry_dt": tr["entry_date"],
                    "exit_dt": dt,
                    "dir": "LONG" if direction == 1 else "SHORT",
                    "entry_a": tr["px_a0"],
                    "entry_b": tr["px_b0"],
                    "exit_a": px_a,
                    "exit_b": px_b,
                    "pnl": pnl,
                    "days_held": (dt - tr["entry_date"]).days,
                })
                
                # Update risk manager equity
                risk_mgr.update_equity(dt, pnl)
                daily_pnl += pnl
            
            # Check if risk manager wants to force exit any positions
            for pair, tr in list(open_trades.items()):
                current_z = z_scores.at[dt, pair] if pair in z_scores.columns else None
                
                if risk_mgr.check_exit(tr, dt, current_z):
                    # Force exit due to risk management
                    px_a_exit = prices.at[dt, tr["ticker_a"]]
                    px_b_exit = prices.at[dt, tr["ticker_b"]]
                    direction = tr["dir"]
                    
                    # Calculate PnL with leverage effect
                    pnl = leverage * direction * (px_a_exit - tr["px_a0"]) * tr["qty"] - leverage * direction * (px_b_exit - tr["px_b0"]) * tr["qty"]
                    
                    # Close position in risk manager
                    risk_mgr.close_trade(pair)
                    
                    # Add to trade log
                    trade_log.append({
                        "pair": pair,
                        "entry_dt": tr["entry_date"],
                        "exit_dt": dt,
                        "dir": "LONG" if direction == 1 else "SHORT",
                        "entry_a": tr["px_a0"],
                        "entry_b": tr["px_b0"],
                        "exit_a": px_a_exit,
                        "exit_b": px_b_exit,
                        "pnl": pnl,
                        "days_held": (dt - tr["entry_date"]).days,
                        "exit_reason": "Risk management"
                    })
                    
                    # Update risk manager equity
                    risk_mgr.update_equity(dt, pnl)
                    daily_pnl += pnl
                    
                    # Remove from open trades
                    open_trades.pop(pair)
                
        # Mark to market unrealized PnL for open trades
        for tr in open_trades.values():
            direction = tr["dir"]
            cur_px_a = prices.at[dt, tr["ticker_a"]]
            cur_px_b = prices.at[dt, tr["ticker_b"]]
            
            # Calculate unrealized PnL with leverage effect
            unrealized_pnl = leverage * direction * (cur_px_a - tr["px_a0"]) * tr["qty"] - leverage * direction * (cur_px_b - tr["px_b0"]) * tr["qty"]
            daily_pnl += unrealized_pnl
        
        equity.append({"Date": dt, "equity": daily_pnl})
        
    # Force closing all open trades at the last day of the data
    if open_trades:
        last_dt = dates[-1]
        for pair_col, tr in open_trades.items():
            px_a = prices.at[last_dt, tr["ticker_a"]]
            px_b = prices.at[last_dt, tr["ticker_b"]]
            direction = tr["dir"]
            
            # Calculate PnL with leverage effect
            pnl = leverage * direction * (px_a - tr["px_a0"]) * tr["qty"] - leverage * direction * (px_b - tr["px_b0"]) * tr["qty"]
            
            trade_log.append({
                "pair": pair_col,
                "entry_dt": tr["entry_date"],
                "exit_dt": last_dt,
                "dir": "LONG" if direction == 1 else "SHORT",
                "entry_a": tr["px_a0"],
                "entry_b": tr["px_b0"],
                "exit_a": px_a,
                "exit_b": px_b,
                "pnl": pnl,
                "days_held": (last_dt - tr["entry_date"]).days,
                "exit_reason": "End of backtest"
            })
            
            # Update risk manager equity
            risk_mgr.update_equity(last_dt, pnl)
            equity[-1]["equity"] += pnl
            
    # Converting the lists into dataframes
    trade_log_df = pd.DataFrame(trade_log)
    equity_df = pd.DataFrame(equity).set_index("Date").cumsum()

    # Summary statistics
    total_pnl = equity_df.iloc[-1]['equity'] if not equity_df.empty else 0
    win_rate = (trade_log_df['pnl'] > 0).mean() * 100 if not trade_log_df.empty else 0
    avg_profit = trade_log_df[trade_log_df['pnl'] > 0]['pnl'].mean() if len(trade_log_df[trade_log_df['pnl'] > 0]) > 0 else 0
    avg_loss = trade_log_df[trade_log_df['pnl'] < 0]['pnl'].mean() if len(trade_log_df[trade_log_df['pnl'] < 0]) > 0 else 0
    profit_factor = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')

    # Calculate CAGR (Compound Annual Growth Rate)
    if len(equity_df) > 0:
        start_date = equity_df.index[0]
        end_date = equity_df.index[-1]
        years = (end_date - start_date).days / 365.25
        start_value = 1000000  # Initial NAV value used when creating RiskManager
        end_value = start_value + total_pnl
        
        if years > 0 and start_value > 0 and end_value > 0:
            cagr = ((end_value / start_value) ** (1 / years) - 1) * 100
        else:
            cagr = 0
    else:
        cagr = 0

    # Calculate max drawdown
    if len(equity_df) > 0:
        cumulative = pd.DataFrame(start_value + equity_df.cumsum())
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min().iloc[0] * 100 if not drawdown.empty else 0
    else:
        max_drawdown = 0

    # Trade statistics
    num_trades = len(trade_log_df)
    trades_per_year = num_trades / years if years > 0 else 0
    avg_trade_pnl = total_pnl / num_trades if num_trades > 0 else 0
    sharpe_ratio = (cagr / (-max_drawdown)) if max_drawdown != 0 else 0

    # Converting the trade and equity dataframes into csv files
    results_dir = f"results/{strategy_name.replace(' ', '_').lower()}"
    os.makedirs(results_dir, exist_ok=True)
    trade_log_df.to_csv(f"{results_dir}/trade_log.csv", index=False)
    equity_df.to_csv(f"{results_dir}/equity.csv", index=True)
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    equity_df.plot()
    plt.title(f'Equity Curve - {strategy_name}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL')
    plt.grid(True)
    plt.savefig(f'{results_dir}/equity_curve.png')
    
    # Print results
    print("\n====== Backtest Results ======")
    print(f"Strategy: {strategy_name}")
    print(f"Number of trades: {num_trades}")
    print(f"Rejected trades: {rejected_trades}")
    print(f"Trades per year: {trades_per_year:.2f}")
    print(f"Total PnL: {total_pnl:.2f}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Average profit: {avg_profit:.2f}")
    print(f"Average loss: {avg_loss:.2f}")
    print(f"Profit factor: {profit_factor:.2f}")
    print(f"CAGR: {cagr:.2f}%")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Return metrics for logging
    metrics = {
        'num_trades': num_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'max_drawdown': max_drawdown,
        'cagr': cagr,
        'sharpe_ratio': sharpe_ratio,
        'profit_factor': profit_factor,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'trades_per_year': trades_per_year,
        'avg_trade_pnl': avg_trade_pnl,
        'rejected_trades': rejected_trades
    }
    
    return metrics

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run statistical arbitrage backtest with configurable parameters')
    
    # Strategy name
    parser.add_argument('--name', type=str, default="Default Strategy", 
                        help='Name for this strategy configuration')
    
    # ML parameters
    parser.add_argument('--ml-threshold', type=float, default=0.4,
                        help='ML probability threshold for trade filtering (default: 0.4)')
    
    # Position sizing and leverage
    parser.add_argument('--leverage', type=float, default=1.0,
                        help='Leverage multiplier (default: 1.0)')
    
    # Risk parameters
    parser.add_argument('--max-drawdown', type=float, default=0.2,
                        help='Maximum drawdown percentage (default: 0.2)')
    parser.add_argument('--max-daily-loss', type=float, default=10000,
                        help='Maximum daily loss in currency units (default: 10000)')
    parser.add_argument('--max-open-trades', type=int, default=5,
                        help='Maximum number of open trades (default: 5)')
    parser.add_argument('--risk-pct', type=float, default=0.01,
                        help='Risk percentage per trade (default: 0.01)')
    parser.add_argument('--stop-loss-pct', type=float, default=0.02,
                        help='Stop loss percentage (default: 0.02)')
    parser.add_argument('--stop-z', type=float, default=3.0,
                        help='Z-score stop level (default: 3.0)')
    parser.add_argument('--max-hold-days', type=int, default=10,
                        help='Maximum holding period in days (default: 10)')
    
    # Logging options
    parser.add_argument('--log-results', action='store_true',
                        help='Log results to the performance log')
    parser.add_argument('--log-file', type=str, default='logs/backtest_results_log.csv',
                        help='Path to the performance log file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Build risk config from arguments
    risk_config = {
        "max_drawdown_pct": args.max_drawdown,
        "max_daily_loss": args.max_daily_loss,
        "max_open_trades": args.max_open_trades,
        "risk_pct": args.risk_pct,
        "stop_loss_pct": args.stop_loss_pct,
        "sizer": "fixed_pct",
        "stop_z": args.stop_z,
        "max_hold_days": args.max_hold_days,
    }
    
    # Run backtest
    metrics = run_backtest(
        risk_config=risk_config,
        ml_threshold=args.ml_threshold,
        leverage=args.leverage,
        strategy_name=args.name
    )
    
    # Log results if requested
    if args.log_results:
        logger = PerformanceLogger(args.log_file)
        logger.log_performance(
            config_name=args.name,
            metrics=metrics,
            risk_config=risk_config,
            ml_threshold=args.ml_threshold,
            leverage=args.leverage
        )
        
        # Show top configurations by CAGR
        print("\nTop Configurations by CAGR:")
        top_configs = logger.compare_configs(metric='cagr', top_n=5)
        if not top_configs.empty:
            print(top_configs)
        else:
            print("No previous configurations found in log.")





