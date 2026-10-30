

import time

import numpy as np

from ..draft import slot_for_pick
from .base import DraftSource, RemotePick


class MockSource(DraftSource):
    """A simulated live draft (DRAFT_PLATFORM=mock): picks arrive on a timer, in snake/linear order, from a noisy
    version of your own rankings. Lets you try auto-sync, 'you drafted X' alerts, team score and pace with no Yahoo."""

    def __init__(self, ranked_ids, teams, rounds, draft_type, my_slot, seconds_per_pick=4.0, seed=1,
                 budget=None, prices=None):
        n = np.arange(len(ranked_ids))
        key = n + np.random.default_rng(seed).normal(0, 2 + 0.12 * n)
        self.order = [ranked_ids[i] for i in np.argsort(key, kind="stable")]
        self.teams, self.total, self.type, self.my_slot = teams, min(teams * rounds, len(ranked_ids)), draft_type, my_slot
        self.seconds, self.t0 = seconds_per_pick, time.monotonic()
        self.sales = None
        if draft_type == "auction":                    # who wins each player, and for how much
            self.sales = self._auction_sales(budget or 260, rounds, prices or {}, np.random.default_rng(seed + 1))

    def _auction_sales(self, budget, rounds, prices, rng):
        """Teams nominate in order 1..N (skipping full rosters). Each sale goes to the nominating team at roughly the
        suggested price (+-30%), never more than that team can legally bid."""
        spent, count, sales, ptr = [0] * self.teams, [0] * self.teams, [], 0
        for pid in self.order[:self.total]:
            for _ in range(self.teams):
                if count[ptr] < rounds:
                    break
                ptr = (ptr + 1) % self.teams
            else:
                break
            open_after = rounds - count[ptr] - 1
            cap = max(budget - spent[ptr] - open_after, 1)
            price = int(min(max(round(prices.get(pid, 1) * rng.uniform(0.7, 1.3)), 1), cap))
            spent[ptr] += price
            count[ptr] += 1
            sales.append((pid, ptr + 1, price))
            ptr = (ptr + 1) % self.teams
        return sales

    async def fetch_picks(self) -> list[RemotePick]:
        k = min(int((time.monotonic() - self.t0) / self.seconds), self.total)
        if self.sales is not None:
            return [RemotePick(i, pid, str(team), cost=price) for i, (pid, team, price) in enumerate(self.sales[:k], 1)]
        return [RemotePick(i, self.order[i - 1], str(slot_for_pick(self.type, self.teams, i))) for i in range(1, k + 1)]

    def team_index(self, platform_team_id: str) -> int:
        return int(platform_team_id)

    def my_team_id(self):
        return self.my_slot




