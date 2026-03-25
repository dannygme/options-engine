from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.amm import AMM
    from src.options import OptionsEngine, OptionType


@dataclass
class AttackResult:
    step: int
    attack_size_x: float
    price_before: float
    price_after: float
    price_impact_pct: float
    swap_cost_y: float
    reversal_revenue_y: float
    net_swap_loss_y: float
    option_payoff: float
    gross_profit: float
    twap_at_settlement: float


class AdversarialAgent:
    def __init__(self, attack_step: int, attack_size_y: float, option_strike: float,
                 option_expiry_step: int, option_collateral: float):
        self.attack_step = attack_step
        self.attack_size_y = attack_size_y
        self.option_strike = option_strike
        self.option_expiry_step = option_expiry_step
        self.option_collateral = option_collateral
        self.active = True
        self.result: AttackResult | None = None
        self.held_x: float = 0.0
        self.price_before: float = 0.0
        self.price_after: float = 0.0
        self.option = None

    def pre_position(self, amm: "AMM", options_engine: "OptionsEngine",
                     option_type, step: int, volatility: float) -> None:
        spot = amm.spot_price
        premium = options_engine.compute_premium(
            option_type, self.option_strike, spot,
            max(self.option_expiry_step - step, 1), volatility
        )
        self.option = options_engine.open_option(
            option_type=option_type,
            strike=self.option_strike,
            expiry_step=self.option_expiry_step,
            collateral=self.option_collateral,
            premium=premium,
            buyer="attacker",
            writer="pool",
        )

    def execute_attack(self, amm: "AMM", step: int) -> float:
        if step != self.attack_step or not self.active:
            return 0.0
        self.price_before = amm.spot_price
        self.held_x = amm.swap_y_for_x(self.attack_size_y)
        self.price_after = amm.spot_price
        return self.held_x

    def reverse_attack(self, amm: "AMM", step: int) -> float:
        if self.held_x <= 0:
            return 0.0
        y_recovered = amm.swap_x_for_y(self.held_x)
        self.held_x = 0.0
        return y_recovered

    def compute_profit(self, option_payoff: float, swap_cost_y: float,
                       reversal_y: float, twap: float, step: int) -> AttackResult:
        net_swap_loss = swap_cost_y - reversal_y
        gross_profit = option_payoff - net_swap_loss
        self.result = AttackResult(
            step=step,
            attack_size_x=self.held_x,
            price_before=self.price_before,
            price_after=self.price_after,
            price_impact_pct=(self.price_after - self.price_before) / self.price_before * 100
            if self.price_before > 0 else 0.0,
            swap_cost_y=swap_cost_y,
            reversal_revenue_y=reversal_y,
            net_swap_loss_y=net_swap_loss,
            option_payoff=option_payoff,
            gross_profit=gross_profit,
            twap_at_settlement=twap,
        )
        return self.result