"""
Position sizing functions for risk management
"""

def fixed_percent_sizer(capital: float, price: float, risk_pct: float = 0.01) -> int:
    """
    Calculate position size based on fixed percentage of capital.
    
    Args:
        capital (float): Total capital available
        price (float): Current price of the asset
        risk_pct (float): Percentage of capital to risk (0.01 = 1%)
        
    Returns:
        int: Number of shares to trade
    """
    # Calculate position size based on risk percentage
    position_value = capital * risk_pct
    
    # Calculate number of shares
    if price > 0:
        shares = int(position_value / price)
    else:
        shares = 0
        
    return shares

def volatility_sizer(capital: float, price: float, volatility: float, risk_pct: float = 0.01) -> int:
    """
    Calculate position size based on volatility.
    
    Args:
        capital (float): Total capital available
        price (float): Current price of the asset
        volatility (float): Volatility measure (e.g., standard deviation)
        risk_pct (float): Percentage of capital to risk (0.01 = 1%)
        
    Returns:
        int: Number of shares to trade
    """
    # Calculate risk amount
    risk_amount = capital * risk_pct
    
    # Scale position size inversely with volatility
    if volatility > 0 and price > 0:
        # Lower volatility = larger position size, higher volatility = smaller position size
        position_value = risk_amount / volatility
        shares = int(position_value / price)
    else:
        shares = 0
        
    return shares
