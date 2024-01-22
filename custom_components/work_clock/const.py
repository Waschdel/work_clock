"""Constants for Work Clock integration."""
import pandas as pd

from homeassistant.const import Platform

# Base component constants
NAME = "Work Clock"
DOMAIN = "work_clock"
VERSION = "0.0.1"

ISSUE_URL = "https://github.com/Waschdel/work_clock/issues"

# Icons
ICON = "mdi:office-building"

# Platforms
PLATFORMS = [
    Platform.SWITCH,
    Platform.SELECT,
    Platform.BUTTON,
    Platform.DATETIME,
    Platform.SENSOR,
    Platform.BINARY_SENSOR,
]

TABLE_STATES = DOMAIN + "_STATES"
TABLE_ENTRIES = DOMAIN + "_ENTRIES"

ATTR_DATETIME = "date_time"
CONF_WORKHOURS = "work_hours"
CONF_START_DATE = "start_date"
CONF_START_FG = "start_fg"
CONF_START_FZ = "start_fz"
DATE_FORMAT = "%Y-%m-%d_%H:%M:%S"
MONTH_FORMAT = "%B %Y"

STATES_SCHEMA = pd.DataFrame(
    {
        "datetime": pd.Series([], dtype="datetime64[s, UTC]"),
        "state": pd.Series([], dtype=bool),
    }
)
STATES_DAY_SCHEMA = pd.DataFrame(
    {
        "date": pd.Series([], dtype="datetime64[s]"),
        "start": pd.Series([], dtype="datetime64[s, UTC]"),
        "end": pd.Series([], dtype="datetime64[s, UTC]"),
        "duration": pd.Series([], dtype="timedelta64[s]"),
    }
)
ENTRIES_SCHEMA = pd.DataFrame(
    {
        "date": pd.Series([], dtype="datetime64[s]"),
        "holiday": pd.Series([], dtype="str"),
        "type": pd.Series([], dtype="str"),
        "start": pd.Series([], dtype="datetime64[s]"),
        "end": pd.Series([], dtype="datetime64[s]"),
        "break": pd.Series([], dtype="timedelta64[s]"),
        "time": pd.Series([], dtype="timedelta64[s]"),
        "time_sum": pd.Series([], dtype=float),
        "time_booked": pd.Series([], dtype=float),
        "src": pd.Series([], dtype=float),
        "tar_hours": pd.Series([], dtype=float),
        "overhours": pd.Series([], dtype=float),
        "time_acc": pd.Series([], dtype="timedelta64[s]"),
        "active": pd.Series([], dtype="timedelta64[s]"),
        "rest": pd.Series([], dtype="timedelta64[s]"),
        "src_start": pd.Series([], dtype="datetime64[s, UTC]"),
        "src_end": pd.Series([], dtype="datetime64[s, UTC]"),
        "calculated": pd.Series([], dtype="bool"),
    }
)

ENTRY_TYPES_K = ["KK", "KA", "KN", "KV"]
ENTRY_TYPES_OFF = ["FG", "U"] + ENTRY_TYPES_K
ENTRY_TYPES_WORK = ["KG", "MA", "D", "DM"]
ENTRY_TYPES = ENTRY_TYPES_OFF + ENTRY_TYPES_WORK

STARTUP_MESSAGE = f"""
-------------------------------------------------------------------
{NAME}
Version: {VERSION}
This is a custom integration!
If you have any issues with this you need to open an issue here:
{ISSUE_URL}
-------------------------------------------------------------------
"""
