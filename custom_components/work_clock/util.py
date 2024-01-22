"""Utils for WorkClock."""
from datetime import datetime
from zoneinfo import ZoneInfo


def utc_now_m() -> datetime:
    """Get now rounded to minutes as UTC."""
    dt = datetime.now(tz=ZoneInfo("UTC"))
    return datetime.fromtimestamp(round(dt.timestamp() / 60) * 60, tz=ZoneInfo("UTC"))
