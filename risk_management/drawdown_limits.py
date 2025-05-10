import pandas as pd

def check_drawdown(equity_curve: pd.Series, max_dd_pct: float) -> bool:
    """
    Checks if current drawdown exceeds allowed threshold.

    Args:
        equity_curve (pd.Series): cumulative equity over time (indexed by date)
        max_dd_pct (float): max allowed drawdown (e.g. 0.2 = 20%)

    Returns:
        bool: True if drawdown breached, False otherwise
    """
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    current_dd = drawdown.iloc[-1]

    return current_dd <= -abs(max_dd_pct)  # drawdown is negative


def check_daily_loss(pnl_by_date: pd.Series, current_date, max_daily_loss: float) -> bool:
    """
    Checks if today's loss exceeds daily loss threshold.

    Args:
        pnl_by_date (pd.Series): PnL per date (indexed by date)
        current_date (datetime or str): date to check
        max_daily_loss (float): max allowed daily loss in ₹

    Returns:
        bool: True if loss breached, False otherwise
    """
    loss_today = pnl_by_date.get(current_date, 0.0)
    return loss_today <= -abs(max_daily_loss)
