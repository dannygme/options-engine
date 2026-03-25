import numpy as np
import pandas as pd
from scipy.stats import norm
from src.adversary import AttackResult
from src.lp_accounting import LPPosition


def black_scholes_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (np.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    call = black_scholes_call(S, K, T, sigma, r)
    return call - S + K * np.exp(-r * T)


def compute_exploitability(attack_result: AttackResult | None) -> dict:
    if attack_result is None:
        return {"attacker_gross_profit": 0.0, "attack_occurred": False}
    return {
        "attack_occurred": True,
        "price_before": attack_result.price_before,
        "price_after": attack_result.price_after,
        "price_impact_pct": attack_result.price_impact_pct,
        "swap_cost_y": attack_result.swap_cost_y,
        "reversal_revenue_y": attack_result.reversal_revenue_y,
        "net_swap_loss_y": attack_result.net_swap_loss_y,
        "option_payoff": attack_result.option_payoff,
        "attacker_gross_profit": attack_result.gross_profit,
        "twap_at_settlement": attack_result.twap_at_settlement,
    }


def compute_pricing_error(simulated_payoff: float, strike: float, entry_spot: float,
                          sigma: float, steps_to_expiry: int) -> dict:
    T = steps_to_expiry / 252
    bs_price = black_scholes_call(entry_spot, strike, T, sigma)
    error = simulated_payoff - bs_price
    pct_error = error / bs_price * 100 if bs_price > 0 else float("inf")
    return {
        "simulated_payoff": simulated_payoff,
        "bs_theoretical": bs_price,
        "absolute_error": error,
        "pct_error": pct_error,
    }


def compute_lp_metrics(position: LPPosition) -> dict:
    return {
        "lp_id": position.lp_id,
        "fees_earned": position.fees_earned,
        "il_fraction": position.il,
        "net_pnl": position.net_pnl,
        "held_duration": (position.exit_step if position.exit_step >= 0 else -1) - position.entry_step,
    }


def compute_market_stability(df: pd.DataFrame) -> dict:
    returns = df["spot_price"].pct_change().dropna()
    return {
        "price_volatility": returns.std(),
        "avg_twap_spot_divergence": (df["twap"] - df["spot_price"]).abs().mean(),
        "max_twap_spot_divergence": (df["twap"] - df["spot_price"]).abs().max(),
        "avg_manipulation_score": df["manip_score"].mean(),
    }