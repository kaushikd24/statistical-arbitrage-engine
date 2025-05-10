from .position_sizing import fixed_percent_sizer, volatility_sizer
from .drawdown_limits import check_drawdown, check_daily_loss
import pandas as pd

class RiskManager:
    def __init__(self, nav_start: float, config: dict):
        self.nav = nav_start
        self.config = config
        self.equity_curve = []
        self.pnl_by_date = {}
        self.open_trades = {}
    
    def update_equity(self, date, pnl: float):
        self.equity_curve.append(pnl)

        # Track PnL by date
        if date in self.pnl_by_date:
            self.pnl_by_date[date] += pnl
        else:
            self.pnl_by_date[date] = pnl

    def check_pre_trade(self, pair: str, date, price: float, spread_vol: float = 1.0) -> dict:
        """
        Check if a new trade is allowed based on risk limits
        
        Returns:
            dict: {"allow": bool, "reason": str, "qty": int}
        """
        # Check if max open trades reached
        if len(self.open_trades) >= self.config.get("max_open_trades", 5):
            return {"allow": False, "reason": "Max open trades reached", "qty": 0}
        
        # Check drawdown limit
        if check_drawdown(self.nav, self.equity_curve, self.config.get("max_drawdown_pct", 0.2)):
            return {"allow": False, "reason": "Drawdown limit hit", "qty": 0}
        
        # Calculate position size
        qty = 0
        
        if self.config.get("sizer") == "volatility":
            qty = volatility_sizer(
                capital=self.nav + sum(self.equity_curve),
                price=price,
                volatility=spread_vol,
                risk_pct=self.config.get("risk_pct", 0.01)
            )
        else:  # Default to fixed percent
            qty = fixed_percent_sizer(
                capital=self.nav + sum(self.equity_curve),
                price=price,
                risk_pct=self.config.get("risk_pct", 0.01)
            )
            
        return {"allow": True, "reason": "Trade allowed", "qty": qty}

    def check_exit(self, trade: dict, date, z_score: float = None) -> bool:
        """
        Check if a trade should be exited based on risk management rules
        
        Args:
            trade (dict): Trade details
            date: Current date
            z_score (float, optional): Current z-score for the pair
            
        Returns:
            bool: True if trade should be exited, False otherwise
        """
        # Check max hold days
        max_days = self.config.get("max_hold_days", 10)
        days_held = (date - trade.get("entry_date")).days
        
        if days_held > max_days:
            return True
            
        # Check z-score exit
        if z_score is not None and self.config.get("stop_z") is not None:
            stop_z = self.config.get("stop_z")
            
            if abs(z_score) > stop_z:
                return True
                
        # Check daily loss
        if date in self.pnl_by_date:
            daily_pnl = self.pnl_by_date[date]
            
            if check_daily_loss(daily_pnl, self.config.get("max_daily_loss", 10000)):
                return True
                
        return False

    def log_trade(self, pair: str, trade_id: int):
        """Log a new trade"""
        self.open_trades[pair] = {"id": trade_id, "pnl": 0}

    def close_trade(self, pair: str):
        """Close a trade"""
        if pair in self.open_trades:
            self.open_trades.pop(pair)
