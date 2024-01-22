"""Utils for WorkClock."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def utc_now_m() -> datetime:
    """Get now rounded to minutes as UTC."""
    dt = datetime.now(tz=ZoneInfo("UTC"))
    return datetime.fromtimestamp(round(dt.timestamp() / 60) * 60, tz=ZoneInfo("UTC"))


def format_timedelta(t: timedelta) -> str:
    """Write timedelta to %H:%M format."""
    m = round(t.total_seconds() / 60)
    return "%02d:%02d" % (int(m / 60), m % 60)


def first_of_month(d: datetime) -> datetime:
    """Get first of the month."""
    return datetime(d.year, d.month, 1)


def last_of_month(d: datetime) -> datetime:
    """Get first of the month."""
    d1 = first_of_month(d)
    if d1.month == 12:
        d1 = d1.replace(year=d1.year + 1, month=1)
    else:
        d1 = d1.replace(month=d1.month + 1)
    return d1 - timedelta(days=1)
