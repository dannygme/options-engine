from typing import List, Tuple


class TWAPOracle:
    def __init__(self, window: int = 20):
        assert window >= 1
        self.window = window

    def compute(self, price_history: List[Tuple[int, float]], current_step: int) -> float:
        if not price_history:
            raise ValueError("No price history available")

        cutoff = current_step - self.window
        relevant = [(t, p) for t, p in price_history if t >= cutoff]

        if len(relevant) < 2:
            return price_history[-1][1]

        times = [t for t, _ in relevant]
        prices = [p for _, p in relevant]

        total_weight = 0.0
        weighted_sum = 0.0
        for i in range(len(times) - 1):
            dt = times[i + 1] - times[i]
            weighted_sum += prices[i] * dt
            total_weight += dt

        dt_final = current_step - times[-1]
        if dt_final > 0:
            weighted_sum += prices[-1] * dt_final
            total_weight += dt_final

        return weighted_sum / total_weight if total_weight > 0 else prices[-1]

    def manipulation_resistance_score(
        self, price_history: List[Tuple[int, float]], current_step: int
    ) -> float:
        spot = price_history[-1][1] if price_history else 1.0
        twap = self.compute(price_history, current_step)
        return twap / spot if spot > 0 else 1.0