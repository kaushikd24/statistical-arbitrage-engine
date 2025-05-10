"""
Parameter Sweep Script for Statistical Arbitrage Strategy

This script runs multiple backtest configurations with different risk parameters,
ML thresholds, and leverage values to find optimal settings.
"""

import itertools
import subprocess
import time
import os
from datetime import datetime
import pandas as pd
from performance_logger import PerformanceLogger

# Make sure the logs directory exists
os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Define parameter ranges to test
parameter_ranges = {
    # Risk tolerance parameters
    "max_drawdown": [0.15, 0.2, 0.25, 0.3, 0.35],
    "max_daily_loss": [5000, 10000, 15000, 20000, 25000],
    "risk_pct": [0.01, 0.015, 0.02, 0.03, 0.04],
    "stop_loss_pct": [0.02, 0.03, 0.04, 0.05],
    
    # Trade filtering parameters
    "ml_threshold": [0.3, 0.4, 0.5, 0.6],
    "leverage": [1.0, 1.5, 2.0, 2.5],
    
    # Position limits
    "max_open_trades": [5, 8, 10, 15],
    
    # Exit parameters
    "stop_z": [3.0, 4.0, 5.0],
    "max_hold_days": [10, 15, 20],
}

# Define parameter combinations to test
# We'll use a smaller subset of combinations to keep runtime reasonable
def generate_test_configs(num_configs=20):
    """Generate a specified number of parameter configurations to test"""
    config_list = []
    
    # Conservative configurations
    config_list.extend([
        {"name": "Conservative_Base", "max_drawdown": 0.15, "max_daily_loss": 5000, 
         "risk_pct": 0.01, "stop_loss_pct": 0.02, "ml_threshold": 0.6, "leverage": 1.0,
         "max_open_trades": 5, "stop_z": 3.0, "max_hold_days": 10},
        
        {"name": "Conservative_Higher_ML", "max_drawdown": 0.15, "max_daily_loss": 5000, 
         "risk_pct": 0.01, "stop_loss_pct": 0.02, "ml_threshold": 0.5, "leverage": 1.0,
         "max_open_trades": 5, "stop_z": 3.0, "max_hold_days": 10},
    ])
    
    # Moderate configurations
    config_list.extend([
        {"name": "Moderate_Base", "max_drawdown": 0.2, "max_daily_loss": 10000, 
         "risk_pct": 0.02, "stop_loss_pct": 0.03, "ml_threshold": 0.5, "leverage": 1.5,
         "max_open_trades": 8, "stop_z": 4.0, "max_hold_days": 15},
        
        {"name": "Moderate_More_Trades", "max_drawdown": 0.2, "max_daily_loss": 10000, 
         "risk_pct": 0.02, "stop_loss_pct": 0.03, "ml_threshold": 0.4, "leverage": 1.5,
         "max_open_trades": 10, "stop_z": 4.0, "max_hold_days": 15},
    ])
    
    # Aggressive configurations
    config_list.extend([
        {"name": "Aggressive_Base", "max_drawdown": 0.25, "max_daily_loss": 15000, 
         "risk_pct": 0.03, "stop_loss_pct": 0.04, "ml_threshold": 0.4, "leverage": 2.0,
         "max_open_trades": 10, "stop_z": 4.0, "max_hold_days": 15},
        
        {"name": "Aggressive_Higher_Risk", "max_drawdown": 0.3, "max_daily_loss": 20000, 
         "risk_pct": 0.04, "stop_loss_pct": 0.05, "ml_threshold": 0.3, "leverage": 2.0,
         "max_open_trades": 15, "stop_z": 5.0, "max_hold_days": 20},
    ])
    
    # Ultra-aggressive configurations
    config_list.extend([
        {"name": "Ultra_Aggressive", "max_drawdown": 0.35, "max_daily_loss": 25000, 
         "risk_pct": 0.04, "stop_loss_pct": 0.05, "ml_threshold": 0.3, "leverage": 2.5,
         "max_open_trades": 15, "stop_z": 5.0, "max_hold_days": 20},
    ])
    
    # Add ML threshold/leverage specific tests
    ml_leverage_combos = list(itertools.product([0.3, 0.4, 0.5], [1.0, 1.5, 2.0, 2.5]))
    for idx, (ml, lev) in enumerate(ml_leverage_combos[:5]):  # Take only first 5 combinations
        config_list.append({
            "name": f"ML{int(ml*100)}_Lev{int(lev*10)}",
            "max_drawdown": 0.25, "max_daily_loss": 15000, 
            "risk_pct": 0.02, "stop_loss_pct": 0.03, "ml_threshold": ml, "leverage": lev,
            "max_open_trades": 10, "stop_z": 4.0, "max_hold_days": 15
        })
    
    # If we still need more configs, add some random combinations
    if len(config_list) < num_configs:
        # Add some combinations focused on specific parameter groupings
        # This is more targeted than purely random combinations
        for _ in range(num_configs - len(config_list)):
            import random
            risk_level = random.choice(["low", "medium", "high"])
            
            if risk_level == "low":
                config = {
                    "name": f"Random_Low_Risk_{_+1}",
                    "max_drawdown": random.choice([0.15, 0.2]),
                    "max_daily_loss": random.choice([5000, 10000]),
                    "risk_pct": random.choice([0.01, 0.015]),
                    "stop_loss_pct": random.choice([0.02, 0.03]),
                    "ml_threshold": random.choice([0.5, 0.6]),
                    "leverage": random.choice([1.0, 1.5]),
                    "max_open_trades": random.choice([5, 8]),
                    "stop_z": random.choice([3.0, 4.0]),
                    "max_hold_days": random.choice([10, 15])
                }
            elif risk_level == "medium":
                config = {
                    "name": f"Random_Medium_Risk_{_+1}",
                    "max_drawdown": random.choice([0.2, 0.25]),
                    "max_daily_loss": random.choice([10000, 15000]),
                    "risk_pct": random.choice([0.02, 0.03]),
                    "stop_loss_pct": random.choice([0.03, 0.04]),
                    "ml_threshold": random.choice([0.4, 0.5]),
                    "leverage": random.choice([1.5, 2.0]),
                    "max_open_trades": random.choice([8, 10]),
                    "stop_z": random.choice([3.0, 4.0]),
                    "max_hold_days": random.choice([15, 20])
                }
            else:  # high risk
                config = {
                    "name": f"Random_High_Risk_{_+1}",
                    "max_drawdown": random.choice([0.25, 0.3, 0.35]),
                    "max_daily_loss": random.choice([15000, 20000, 25000]),
                    "risk_pct": random.choice([0.03, 0.04]),
                    "stop_loss_pct": random.choice([0.04, 0.05]),
                    "ml_threshold": random.choice([0.3, 0.4]),
                    "leverage": random.choice([2.0, 2.5]),
                    "max_open_trades": random.choice([10, 15]),
                    "stop_z": random.choice([4.0, 5.0]),
                    "max_hold_days": random.choice([15, 20])
                }
            
            config_list.append(config)
    
    return config_list[:num_configs]  # Limit to the requested number of configs

def run_parameter_sweep(configs, log_file="logs/backtest_results_log.csv"):
    """Run backtest with each parameter configuration"""
    print(f"Starting parameter sweep with {len(configs)} configurations")
    print(f"Results will be logged to {log_file}")
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running configuration: {config['name']}")
        
        cmd = [
            "python3", "backtest_engine.py",
            "--name", config["name"],
            "--ml-threshold", str(config["ml_threshold"]),
            "--leverage", str(config["leverage"]),
            "--max-drawdown", str(config["max_drawdown"]),
            "--max-daily-loss", str(config["max_daily_loss"]),
            "--max-open-trades", str(config["max_open_trades"]),
            "--risk-pct", str(config["risk_pct"]),
            "--stop-loss-pct", str(config["stop_loss_pct"]),
            "--stop-z", str(config["stop_z"]),
            "--max-hold-days", str(config["max_hold_days"]),
            "--log-results",
            "--log-file", log_file
        ]
        
        try:
            start_time = time.time()
            subprocess.run(cmd, check=True)
            duration = time.time() - start_time
            print(f"Completed in {duration:.1f} seconds")
        except subprocess.CalledProcessError as e:
            print(f"Error running configuration {config['name']}: {e}")
    
    print("\nParameter sweep complete!")

def analyze_results(log_file="logs/backtest_results_log.csv"):
    """Analyze and visualize backtest results"""
    logger = PerformanceLogger(log_file)
    df = logger.get_all_logs()
    
    if df.empty:
        print("No results found in log file")
        return
    
    print("\n===== Parameter Sweep Results =====")
    
    # Top configurations by different metrics
    metrics = ['cagr', 'sharpe_ratio', 'profit_factor', 'win_rate', 'total_pnl']
    
    for metric in metrics:
        print(f"\nTop 5 configurations by {metric.upper()}:")
        top = logger.compare_configs(metric=metric, top_n=5)
        if not top.empty:
            print(top)
        
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    for metric in metrics:
        plt = logger.plot_comparison(metric=metric, top_n=5)
        if plt:
            plt.close()
    
    # Create a correlation heatmap of parameters vs. metrics
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        # Extract parameters from risk_config JSON
        param_cols = ['ml_threshold', 'leverage', 'max_drawdown', 'risk_pct', 
                     'stop_loss_pct', 'max_open_trades']
        
        metric_cols = ['cagr', 'sharpe_ratio', 'profit_factor', 'win_rate', 
                      'total_pnl', 'num_trades']
        
        # Create parameter-metric correlation matrix
        corr_data = pd.DataFrame()
        
        # Add parameters
        corr_data['ml_threshold'] = df['ml_threshold']
        corr_data['leverage'] = df['leverage']
        
        # Extract params from risk_config
        for param in ['max_drawdown_pct', 'risk_pct', 'stop_loss_pct', 'max_open_trades']:
            param_short = param.replace('_pct', '').replace('max_', '')
            corr_data[param_short] = df['risk_config_dict'].apply(
                lambda x: x.get(param, None) if isinstance(x, dict) else None
            )
        
        # Add metrics
        for col in metric_cols:
            corr_data[col] = df[col]
        
        # Calculate correlation matrix
        corr_matrix = corr_data.corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.zeros_like(corr_matrix, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                   vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
        
        plt.title('Parameter-Metric Correlation Matrix')
        plt.tight_layout()
        plt.savefig('parameter_correlation.png')
        print("Generated parameter correlation heatmap: parameter_correlation.png")
        
    except Exception as e:
        print(f"Error generating correlation matrix: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run parameter sweep for statistical arbitrage backtest')
    parser.add_argument('--configs', type=int, default=10, 
                        help='Number of configurations to test (default: 10)')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze existing results without running new backtests')
    parser.add_argument('--log-file', type=str, default='logs/backtest_results_log.csv',
                        help='Path to the performance log file')
    
    args = parser.parse_args()
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Parameter sweep started at: {timestamp}")
    
    if not args.analyze_only:
        # Generate configurations
        configs = generate_test_configs(num_configs=args.configs)
        
        # Run parameter sweep
        run_parameter_sweep(configs, log_file=args.log_file)
    
    # Analyze results
    analyze_results(log_file=args.log_file) 