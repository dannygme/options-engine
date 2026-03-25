import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class AMMState:
    reserve_x: float
    reserve_y: float
    total_lp_shares: float
    fee_rate: float = 0.003
    price_history: List[Tuple[int, float]] = field(default_factory=list)
    fee_accumulated: float = 0.0


class AMM:
    def __init__(self, reserve_x: float, reserve_y: float, fee_rate: float = 0.003):
        assert reserve_x > 0 and reserve_y > 0
        self.state = AMMState(
            reserve_x=reserve_x,
            reserve_y=reserve_y,
            total_lp_shares=np.sqrt(reserve_x * reserve_y),
            fee_rate=fee_rate,
        )
        self._k = reserve_x * reserve_y

    @property
    def spot_price(self) -> float:
        return self.state.reserve_y / self.state.reserve_x

    @property
    def k(self) -> float:
        return self._k

    def record_price(self, timestep: int) -> None:
        self.state.price_history.append((timestep, self.spot_price))

    def swap_x_for_y(self, dx: float) -> float:
        assert dx > 0
        dx_after_fee = dx * (1 - self.state.fee_rate)
        fee = dx * self.state.fee_rate
        new_x = self.state.reserve_x + dx_after_fee
        new_y = self._k / new_x
        dy = self.state.reserve_y - new_y
        assert dy > 0
        self.state.reserve_x += dx
        self.state.reserve_y -= dy
        self.state.fee_accumulated += fee * self.spot_price
        self._k = self.state.reserve_x * self.state.reserve_y
        return dy

    def swap_y_for_x(self, dy: float) -> float:
        assert dy > 0
        dy_after_fee = dy * (1 - self.state.fee_rate)
        fee = dy * self.state.fee_rate
        new_y = self.state.reserve_y + dy_after_fee
        new_x = self._k / new_y
        dx = self.state.reserve_x - new_x
        assert dx > 0
        self.state.reserve_y += dy
        self.state.reserve_x -= dx
        self.state.fee_accumulated += fee
        self._k = self.state.reserve_x * self.state.reserve_y
        return dx

    def add_liquidity(self, dx: float, dy: float = None) -> float:
        if dy is None:
            dy = dx * self.spot_price
        ratio = dx / self.state.reserve_x
        shares = ratio * self.state.total_lp_shares
        self.state.reserve_x += dx
        self.state.reserve_y += dy
        self.state.total_lp_shares += shares
        self._k = self.state.reserve_x * self.state.reserve_y
        return shares

    def remove_liquidity(self, shares: float) -> Tuple[float, float]:
        assert shares <= self.state.total_lp_shares
        ratio = shares / self.state.total_lp_shares
        dx = ratio * self.state.reserve_x
        dy = ratio * self.state.reserve_y
        self.state.reserve_x -= dx
        self.state.reserve_y -= dy
        self.state.total_lp_shares -= shares
        self._k = self.state.reserve_x * self.state.reserve_y
        return dx, dy