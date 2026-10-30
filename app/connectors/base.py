
"""Every draft platform implements this. The engine only ever sees RemotePick."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


class SourceError(Exception):
    """A problem with the platform connection, already worded for a person (shown as-is on the page)."""


@dataclass
class RemotePick:
    pick_number: int
    platform_player_id: str
    platform_team_id: str
    player_name: str = ""          # optional: lets the engine match by name when an id isn't in your crosswalk
    cost: int | None = None        # auction price in whole dollars, when the platform reports one


class DraftSource(ABC):
    @abstractmethod
    async def fetch_picks(self) -> list[RemotePick]:
        """Return ALL picks so far, in order. Re-fetching the full list (not deltas) makes
        commissioner undos/corrections self-healing."""

    @abstractmethod
    def team_index(self, platform_team_id: str) -> int:
        """Map the platform's team id to your 1..N team_id."""

    def my_team_id(self) -> int | None:
        return None

    def resolve_names(self, platform_player_ids: list[str]) -> dict[str, str]:
        """Optional: platform player id -> full name, for picks whose id isn't in your crosswalk and whose pick
        data carried no name. Blocking call (the sync loop runs it in a thread). Return {} if unsupported."""
        return {}


def parse_cost(v) -> int | None:
    """An auction price from whatever the platform sent ('23', 23, 23.0, '$23'). None if it isn't a usable price."""
    if v is None or isinstance(v, bool):
        return None
    try:
        n = float(str(v).strip().lstrip("$").replace(",", ""))
    except ValueError:
        return None
    return int(n) if n == int(n) and n >= 1 else None


def dense_index(team_numbers: list[int]):
    """Platforms number teams by internal id, which can have gaps (a team was removed: 1,2,4,5...).
    Returns f(platform_number) -> 1..N so your team ids always sit inside 1..teams."""
    order = {n: i + 1 for i, n in enumerate(sorted(set(team_numbers)))}
    return lambda n: order.get(int(n), int(n))


