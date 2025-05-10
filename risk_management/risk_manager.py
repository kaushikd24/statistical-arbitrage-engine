from position_sizing import fixed_percent_sizer, volatility_sizer
from drawdown_limits import check_drawdown, check_daily_loss
import pandas as pd

class RiskManager:
    def __init__(self, nav_start: float, config: dict):
        self.nav = nav_start
        self.config = config
        self.equity_curve = [nav_start]
        self.pnl_by_date = {}
        self.open_trades = {}
    
    def update_equity(self, date, pnl):
        self.nav += pnl
        self.equity_curve.append(self.nav)

        # Track PnL by date
        if date not in self.pnl_by_date:
            self.pnl_by_date[date] = 0.0
        self.pnl_by_date[date] += pnl

    def check_pre_trade(self, pair, date, price, spread_vol):
        max_dd_pct = self.config["max_drawdown_pct"]
        max_daily_loss = self.config["max_daily_loss"]
        max_open_trades = self.config["max_open_trades"]

        equity_series = pd.Series(self.equity_curve)

        if check_drawdown(equity_series, max_dd_pct):
            return {"allow": False, "reason": "Drawdown limit hit", "qty": 0}

        if check_daily_loss(pd.Series(self.pnl_by_date), date, max_daily_loss):
            return {"allow": False, "reason": "Daily loss limit hit", "qty": 0}

        if len(self.open_trades) >= max_open_trades:
            return {"allow": False, "reason": "Max open trades exceeded", "qty": 0}

        # Position sizing
        if self.config["sizer"] == "fixed_pct":
            qty = fixed_percent_sizer(
                nav=self.nav,
                risk_pct=self.config["risk_pct"],
                stop_loss_pct=self.config["stop_loss_pct"],
                price=price
            )
        elif self.config["sizer"] == "volatility":
            qty = volatility_sizer(
                vol=spread_vol,
                risk_amount=self.nav * self.config["risk_pct"]
            )
        else:
            qty = 0

        return {"allow": True, "reason": "Pass", "qty": qty}

    def check_exit(self, trade, current_date, current_z=None):
        # Stop loss based on z-score
        if self.config.get("stop_z") and abs(current_z) > self.config["stop_z"]:
            return True

        # Time stop
        hold_days = (current_date - trade["entry_date"]).days
        if hold_days >= self.config["max_hold_days"]:
            return True

        return False

    def log_trade(self, pair, trade_id):
        self.open_trades[pair] = trade_id

    def close_trade(self, pair):
        if pair in self.open_trades:
            del self.open_trades[pair]
