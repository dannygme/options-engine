from enum import Enum
from dataclasses import dataclass
from typing import Optional
import numpy as np


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


class OptionStatus(Enum):
    OPEN = "open"
    EXPIRED = "expired"
    SETTLED = "settled"


@dataclass
class Option:
    option_id: int
    option_type: OptionType
    strike: float
    expiry_step: int
    collateral: float
    premium: float
    buyer: str
    writer: str
    notional: float = 1.0
    status: OptionStatus = OptionStatus.OPEN
    settlement_price: Optional[float] = None
    payoff: float = 0.0


class OptionsEngine:
    def __init__(self, twap_oracle):
        self.oracle = twap_oracle
        self.options: list[Option] = []
        self._next_id = 0

    def _next_option_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def compute_premium(self, option_type: OptionType, strike: float, spot: float,
                        steps_to_expiry: int, volatility: float) -> float:
        intrinsic = max(0.0, spot - strike) if option_type == OptionType.CALL else max(0.0, strike - spot)
        time_value = volatility * spot * np.sqrt(max(steps_to_expiry, 1) / 252)
        return intrinsic + time_value

    def open_option(self, option_type: OptionType, strike: float, expiry_step: int,
                    collateral: float, premium: float, buyer: str, writer: str) -> Option:
        notional = collateral / strike if strike > 0 else 1.0
        opt = Option(
            option_id=self._next_option_id(),
            option_type=option_type,
            strike=strike,
            expiry_step=expiry_step,
            collateral=collateral,
            premium=premium,
            buyer=buyer,
            writer=writer,
            notional=notional,
        )
        self.options.append(opt)
        return opt

    def settle_option(self, option: Option, settlement_twap: float) -> float:
        assert option.status == OptionStatus.EXPIRED
        option.settlement_price = settlement_twap
        if option.option_type == OptionType.CALL:
            raw_payoff = max(0.0, settlement_twap - option.strike)
        else:
            raw_payoff = max(0.0, option.strike - settlement_twap)
        payoff = min(raw_payoff * option.notional, option.collateral)
        option.payoff = payoff
        option.status = OptionStatus.SETTLED
        return payoff

    def expire_options(self, current_step: int, price_history, twap_window: int) -> list[Option]:
        settled = []
        for opt in self.options:
            if opt.status == OptionStatus.OPEN and opt.expiry_step <= current_step:
                opt.status = OptionStatus.EXPIRED
                twap = self.oracle.compute(price_history, current_step)
                self.settle_option(opt, twap)
                settled.append(opt)
        return settled