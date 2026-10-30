
from .base import DraftSource, RemotePick


class ManualSource(DraftSource):
    """No live feed: picks come only from POST /draft/pick. Use this for testing."""
    async def fetch_picks(self) -> list[RemotePick]:
        return []

    def team_index(self, platform_team_id: str) -> int:
        return int(platform_team_id)



