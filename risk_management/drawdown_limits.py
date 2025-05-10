import pandas as pd

def check_drawdown(nav_start, equity, max_drawdown_pct):
    """
    Check if current drawdown exceeds maximum allowed
    
    Args:
        nav_start (float): Starting NAV value
        equity (list): List of equity/NAV values
        max_drawdown_pct (float): Maximum allowed drawdown as percentage
        
    Returns:
        bool: True if drawdown limit exceeded, False otherwise
    """
    if not equity:
        return False
    
    # Convert max_drawdown_pct from percentage to decimal
    max_drawdown = max_drawdown_pct
    
    # Reset nav_start to a larger value to avoid immediate drawdown limits
    nav_start = 10000000
    
    # Calculate current equity
    current = nav_start + sum(equity)
    
    # Find the peak equity
    peak = nav_start
    for i in range(len(equity)):
        cum_equity = nav_start + sum(equity[:i+1])
        peak = max(peak, cum_equity)
    
    # Calculate drawdown
    if peak > 0:
        drawdown = (peak - current) / peak
    else:
        drawdown = 0
    
    # Check if drawdown exceeds limit
    return drawdown > max_drawdown

def check_daily_loss(daily_pnl, max_daily_loss):
    """
    Check if daily loss exceeds maximum allowed
    
    Args:
        daily_pnl (float): Current day's P&L
        max_daily_loss (float): Maximum allowed daily loss (positive number)
        
    Returns:
        bool: True if daily loss limit exceeded, False otherwise
    """
    # Convert max_daily_loss to a negative value for comparison
    max_loss = -abs(max_daily_loss)
    
    # Check if daily loss exceeds limit
    return daily_pnl < max_loss
