import numpy as np
import pandas as pd
from src.amm import AMM
from src.twap import TWAPOracle
from src.options import OptionsEngine, OptionType
from src.lp_accounting import LPAccounting
from src.adversary import AdversarialAgent


class SimulationConfig:
    def __init__(
        self,
        n_steps: int = 100,
        seed: int = 42,
        initial_reserve_x: float = 1000.0,
        initial_reserve_y: float = 500_000.0,
        fee_rate: float = 0.003,
        twap_window: int = 20,
        n_traders: int = 5,
        trader_vol: float = 0.02,
        attack_step: int = 50,
        attack_size_y: float = 200_000.0,
        option_strike_offset: float = 0.05,
        option_expiry_offset: int = 20,
        lp_deposit_x: float = 100.0,
        run_attack: bool = True,
    ):
        self.n_steps = n_steps
        self.seed = seed
        self.initial_reserve_x = initial_reserve_x
        self.initial_reserve_y = initial_reserve_y
        self.fee_rate = fee_rate
        self.twap_window = twap_window
        self.n_traders = n_traders
        self.trader_vol = trader_vol
        self.attack_step = attack_step
        self.attack_size_y = attack_size_y
        self.option_strike_offset = option_strike_offset
        self.option_expiry_offset = option_expiry_offset
        self.lp_deposit_x = lp_deposit_x
        self.run_attack = run_attack

        if self.attack_step >= self.n_steps:
            self.attack_step = int(self.n_steps * 0.4)
        self.expiry_step = min(
            self.attack_step + 1,
            self.n_steps - 1,
        )


class Simulation:
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

        self.amm = AMM(config.initial_reserve_x, config.initial_reserve_y, config.fee_rate)
        self.twap = TWAPOracle(window=config.twap_window)
        self.options_engine = OptionsEngine(self.twap)
        self.lp_accounting = LPAccounting()

        initial_price = self.amm.spot_price
        lp_dy = config.lp_deposit_x * initial_price
        shares = self.amm.add_liquidity(config.lp_deposit_x, lp_dy)
        pool_share = shares / self.amm.state.total_lp_shares
        self.lp_accounting.open_position(
            lp_id="lp_0",
            step=0,
            price=initial_price,
            shares=shares,
            pool_share_fraction=pool_share,
            initial_value_y=2 * lp_dy,
        )

        if config.run_attack:
            self.attacker = AdversarialAgent(
                attack_step=config.attack_step,
                attack_size_y=config.attack_size_y,
                option_strike=0.0,
                option_expiry_step=config.expiry_step,
                option_collateral=config.attack_size_y * 2.0,
            )
        else:
            self.attacker = None

        self.log: list[dict] = []
        self.attack_reversal_y: float = 0.0
        self.attack_swap_cost_y: float = 0.0

    def _pool_value_y(self) -> float:
        return self.amm.state.reserve_y + self.amm.state.reserve_x * self.amm.spot_price

    def _trader_step(self) -> None:
        for _ in range(self.cfg.n_traders):
            size = abs(self.rng.normal(0, self.cfg.trader_vol) * self.amm.state.reserve_y * 0.01)
            direction = self.rng.choice(["buy", "sell"])
            try:
                if direction == "buy":
                    self.amm.swap_y_for_x(max(size, 1.0))
                else:
                    self.amm.swap_x_for_y(max(size * self.amm.spot_price, 0.01))
            except AssertionError:
                pass

    def run(self) -> pd.DataFrame:
        pre_position_step = max(1, self.cfg.attack_step - 15)
        pre_positioned = False

        for step in range(self.cfg.n_steps):

            if self.attacker and not pre_positioned and step == pre_position_step:
                spot = self.amm.spot_price
                self.attacker.option_strike = spot * (1 + self.cfg.option_strike_offset)
                self.attacker.option_expiry_step = self.cfg.expiry_step
                self.attacker.pre_position(
                    self.amm, self.options_engine,
                    OptionType.CALL, step, volatility=0.3,
                )
                pre_positioned = True

            if self.attacker and step == self.attacker.attack_step:
                self.attack_swap_cost_y = self.cfg.attack_size_y
                self.attacker.execute_attack(self.amm, step)

            if self.attacker and step == self.attacker.attack_step + 2:
                self.attack_reversal_y = self.attacker.reverse_attack(self.amm, step)

            self._trader_step()

            self.amm.record_price(step)

            settled = self.options_engine.expire_options(
                step, self.amm.state.price_history, self.cfg.twap_window
            )

            fee_share = (
                self.lp_accounting.positions["lp_0"].pool_share_fraction
                * self.amm.state.fee_accumulated
            )
            self.lp_accounting.accrue_fees("lp_0", fee_share)

            lp_snap = self.lp_accounting.snapshot(
                "lp_0", step, self.amm.spot_price, self._pool_value_y()
            )

            twap = self.twap.compute(self.amm.state.price_history, step)
            manip_score = self.twap.manipulation_resistance_score(
                self.amm.state.price_history, step
            )

            self.log.append({
                "step": step,
                "spot_price": self.amm.spot_price,
                "twap": twap,
                "manip_score": manip_score,
                "reserve_x": self.amm.state.reserve_x,
                "reserve_y": self.amm.state.reserve_y,
                "lp_net_pnl": lp_snap["net_pnl"],
                "lp_il_value": lp_snap["il_value_y"],
                "lp_fees": lp_snap["fees_earned"],
                "options_settled_this_step": len(settled),
            })

        if self.attacker and self.attacker.result is None and pre_positioned:
            option_payoff = self.attacker.option.payoff if self.attacker.option else 0.0
            twap_final = self.twap.compute(self.amm.state.price_history, self.cfg.n_steps - 1)
            self.attacker.compute_profit(
                option_payoff,
                self.attack_swap_cost_y,
                self.attack_reversal_y,
                twap_final,
                self.cfg.n_steps - 1,
            )

        return pd.DataFrame(self.log)