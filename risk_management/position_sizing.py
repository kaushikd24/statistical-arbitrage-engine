def fixed_percent_sizer(nav, risk_pct, stop_loss_pct, price):
    """
    Position sizing based on fixed % of NAV and stop-loss buffer.

    Args:
        nav (float): current net asset value (₹)
        risk_pct (float): risk per trade, e.g. 0.01 = 1%
        stop_loss_pct (float): expected worst-case move, e.g. 0.02 = 2%
        price (float): current price of the leg you're sizing

    Returns:
        int: number of units to trade
    """
    risk_amount = nav * risk_pct
    position_value = risk_amount / stop_loss_pct
    size = position_value / price
    return int(size)


def volatility_sizer(vol, risk_amount):
    """
    Position sizing based on volatility of spread.

    Args:
        vol (float): current spread volatility (1*(sigma))
        risk_amount (float): capital you're willing to risk

    Returns:
        int: number of spread units to trade
    """
    if vol == 0:
        return 0
    size = risk_amount / vol
    return int(size)
