from dataclasses import dataclass, field
from typing import List


@dataclass
class LPPosition:
    lp_id: str
    entry_step: int
    entry_price: float
    shares: float
    pool_share_fraction: float
    initial_value_y: float
    fees_earned: float = 0.0
    exit_step: int = -1
    exit_price: float = -1.0
    il: float = 0.0
    net_pnl: float = 0.0
    history: List[dict] = field(default_factory=list)


class LPAccounting:
    def __init__(self):
        self.positions: dict[str, LPPosition] = {}

    def open_position(self, lp_id: str, step: int, price: float,
                      shares: float, pool_share_fraction: float,
                      initial_value_y: float) -> LPPosition:
        pos = LPPosition(
            lp_id=lp_id,
            entry_step=step,
            entry_price=price,
            shares=shares,
            pool_share_fraction=pool_share_fraction,
            initial_value_y=initial_value_y,
        )
        self.positions[lp_id] = pos
        return pos

    def accrue_fees(self, lp_id: str, fee_amount_y: float) -> None:
        if lp_id in self.positions:
            self.positions[lp_id].fees_earned += fee_amount_y

    def compute_il(self, entry_price: float, current_price: float) -> float:
        if entry_price <= 0:
            return 0.0
        r = current_price / entry_price
        return 2 * (r ** 0.5) / (1 + r) - 1

    def snapshot(self, lp_id: str, step: int, current_price: float,
                 current_pool_value_y: float) -> dict:
        pos = self.positions[lp_id]
        il = self.compute_il(pos.entry_price, current_price)
        lp_value = pos.pool_share_fraction * current_pool_value_y
        il_value = lp_value - pos.initial_value_y
        net_pnl = pos.fees_earned + il_value
        snap = {
            "step": step,
            "price": current_price,
            "il_fraction": il,
            "il_value_y": il_value,
            "fees_earned": pos.fees_earned,
            "net_pnl": net_pnl,
            "lp_value": lp_value,
        }
        pos.history.append(snap)
        pos.il = il
        pos.net_pnl = net_pnl
        return snap

    def close_position(self, lp_id: str, step: int, exit_price: float,
                       current_pool_value_y: float) -> LPPosition:
        pos = self.positions[lp_id]
        self.snapshot(lp_id, step, exit_price, current_pool_value_y)
        pos.exit_step = step
        pos.exit_price = exit_price
        return pos