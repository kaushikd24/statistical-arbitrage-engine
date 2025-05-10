"""
Performance Logger for Statistical Arbitrage Backtests

This module logs performance metrics from different backtest runs with varying risk parameters.
It helps track and compare strategy performance across different configurations.
"""

import pandas as pd
import numpy as np
import os
import datetime
import json
from pathlib import Path

class PerformanceLogger:
    def __init__(self, log_file='backtest_results_log.csv'):
        """Initialize the performance logger."""
        self.log_file = log_file
        self.ensure_log_file_exists()
        
    def ensure_log_file_exists(self):
        """Create the log file with headers if it doesn't exist."""
        if not os.path.exists(self.log_file):
            # Create directory if needed
            Path(os.path.dirname(self.log_file)).mkdir(parents=True, exist_ok=True)
            
            # Create file with headers
            headers = [
                'timestamp', 'run_id', 'config_name', 'num_trades', 'win_rate',
                'total_pnl', 'max_drawdown', 'cagr', 'sharpe_ratio', 'profit_factor',
                'avg_profit', 'avg_loss', 'ml_threshold', 'leverage', 'risk_config'
            ]
            pd.DataFrame(columns=headers).to_csv(self.log_file, index=False)
            print(f"Created new performance log file: {self.log_file}")
    
    def log_performance(self, config_name, metrics, risk_config, ml_threshold, leverage):
        """
        Log performance metrics from a backtest run.
        
        Args:
            config_name (str): Name/description of this configuration
            metrics (dict): Performance metrics (num_trades, win_rate, etc.)
            risk_config (dict): Risk management configuration used
            ml_threshold (float): ML probability threshold used
            leverage (float): Leverage multiplier used
        """
        # Create log entry
        log_entry = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'run_id': datetime.datetime.now().strftime('%Y%m%d%H%M%S'),
            'config_name': config_name,
            'num_trades': metrics.get('num_trades', 0),
            'win_rate': metrics.get('win_rate', 0),
            'total_pnl': metrics.get('total_pnl', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'cagr': metrics.get('cagr', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'profit_factor': metrics.get('profit_factor', 0),
            'avg_profit': metrics.get('avg_profit', 0),
            'avg_loss': metrics.get('avg_loss', 0),
            'ml_threshold': ml_threshold,
            'leverage': leverage,
            'risk_config': json.dumps(risk_config)
        }
        
        # Append to log file
        log_df = pd.DataFrame([log_entry])
        log_df.to_csv(self.log_file, mode='a', header=False, index=False)
        print(f"Logged performance metrics for '{config_name}'")
        
    def get_all_logs(self):
        """Return all logged performance metrics as a DataFrame."""
        if os.path.exists(self.log_file):
            df = pd.read_csv(self.log_file)
            # Convert risk_config from JSON string back to dict
            df['risk_config_dict'] = df['risk_config'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
            return df
        return pd.DataFrame()
    
    def compare_configs(self, metric='cagr', top_n=5):
        """
        Compare configurations based on a specific metric.
        
        Args:
            metric (str): Metric to compare (cagr, sharpe_ratio, etc.)
            top_n (int): Number of top configurations to return
            
        Returns:
            DataFrame of top configurations by the metric
        """
        df = self.get_all_logs()
        if df.empty:
            return pd.DataFrame()
        
        # Sort by the specified metric
        sorted_df = df.sort_values(by=metric, ascending=False).head(top_n)
        
        # Select columns for display
        display_cols = ['config_name', 'run_id', metric, 'num_trades', 
                       'win_rate', 'total_pnl', 'sharpe_ratio', 'ml_threshold', 'leverage']
        
        return sorted_df[display_cols]
    
    def plot_comparison(self, metric='cagr', top_n=5):
        """
        Plot comparison of different configurations based on specified metric.
        
        Args:
            metric (str): Metric to compare
            top_n (int): Number of top configurations to plot
            
        Returns:
            matplotlib.pyplot figure
        """
        import matplotlib.pyplot as plt
        
        top_configs = self.compare_configs(metric=metric, top_n=5)
        if top_configs.empty or len(top_configs) < 2:
            print(f"Not enough data available for plotting {metric}")
            return None
        
        # Create bar chart
        plt.figure(figsize=(12, 6))
        bars = plt.bar(top_configs['config_name'].values.tolist(), top_configs[metric].values.tolist())
        
        # Add values above bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.title(f'Top {top_n} Configurations by {metric.upper()}')
        plt.xlabel('Configuration')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f'top_{top_n}_{metric}_comparison.png')
        return plt


if __name__ == "__main__":
    # Example usage
    logger = PerformanceLogger('logs/backtest_results_log.csv')
    
    # Example of how to extract results from a finished backtest
    sample_metrics = {
        'num_trades': 120,
        'win_rate': 58.3,
        'total_pnl': 45000,
        'max_drawdown': -15.2,
        'cagr': 12.5,
        'sharpe_ratio': 1.8,
        'profit_factor': 2.1,
        'avg_profit': 1250,
        'avg_loss': -750
    }
    
    sample_risk_config = {
        'max_drawdown_pct': 0.25,
        'max_daily_loss': 15000,
        'max_open_trades': 10,
        'risk_pct': 0.02,
        'stop_loss_pct': 0.03,
        'sizer': 'fixed_pct',
        'stop_z': 4.0,
        'max_hold_days': 15
    }
    
    # Log sample performance
    logger.log_performance("Sample Conservative Strategy", 
                           sample_metrics, 
                           sample_risk_config, 
                           ml_threshold=0.5, 
                           leverage=1.0)
    
    # Get and display all logged results
    all_logs = logger.get_all_logs()
    print("\nAll Logged Results:")
    print(all_logs[['config_name', 'total_pnl', 'cagr', 'sharpe_ratio']].head())
    
    # Compare based on CAGR
    print("\nTop Configurations by CAGR:")
    print(logger.compare_configs(metric='cagr', top_n=3)) 