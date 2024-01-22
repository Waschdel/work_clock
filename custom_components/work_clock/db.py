"""Database functions for work clock."""
from datetime import date, datetime, timedelta
import fnmatch
import logging
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy.engine import Engine, Result
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.sql import text

from homeassistant.components.recorder import SupportedDialect, get_instance
from homeassistant.components.sql import redact_credentials
from homeassistant.components.sql.sensor import (
    _async_get_or_init_domain_data,
    _generate_lambda_stmt,
    _validate_and_get_session_maker_for_db_url,
)
from homeassistant.components.switch import ENTITY_ID_FORMAT
from homeassistant.const import CONF_NAME, CONF_TIME_ZONE
from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import ConfigType

from .const import (
    CONF_START_DATE,
    CONF_START_FG,
    CONF_START_FZ,
    ENTRIES_SCHEMA,
    ENTRY_TYPES,
    ENTRY_TYPES_OFF,
    ENTRY_TYPES_WORK,
    STATES_DAY_SCHEMA,
    STATES_SCHEMA,
    TABLE_ENTRIES,
    TABLE_STATES,
)
from .util import utc_now_m

_LOGGER = logging.getLogger(__name__)


class WorkClockDbClient:
    """WorkClock states class."""

    reload: bool = True
    states: pd.DataFrame = STATES_SCHEMA
    states_day: pd.DataFrame = STATES_DAY_SCHEMA
    entries: pd.DataFrame = ENTRIES_SCHEMA
    is_on: bool | None = None
    selected_state: datetime | None = None
    new_state: datetime | None = None
    selected_month: datetime | None = None
    selected_entry: int | None = None
    new_type: str | None = None
    new_t_start: datetime | None = None
    new_t_end: datetime | None = None

    def __init__(
        self,
        hass: HomeAssistant,
        config: ConfigType,
        sessmaker: scoped_session,
        use_database_executor: bool,
    ) -> None:
        """WorkClock states class."""
        self.name: str = config.get(CONF_NAME)
        self.entity_id: str = ENTITY_ID_FORMAT.format(self.name)
        self.tz: ZoneInfo = ZoneInfo(config.get(CONF_TIME_ZONE))
        self.start_date: pd.Timestamp = pd.Timestamp(config.get(CONF_START_DATE))
        self.start_fg: float = float(config.get(CONF_START_FG))
        self.start_fz: float = float(config.get(CONF_START_FZ))
        self.hass: HomeAssistant = hass
        self.sessionmaker: scoped_session = sessmaker
        self._use_database_executor: bool = use_database_executor

    async def _async_exec_query(self, query: str) -> Result | None:
        return await async_exec_query(
            self.hass, self.sessionmaker, self._use_database_executor, query
        )

    def get_entries_month(self) -> pd.DataFrame | None:
        """Get entries of selected month."""
        if self.selected_month is None:
            return
        y = self.selected_month.year
        m = self.selected_month.month
        mask = (self.entries["date"].dt.year == y) & (
            self.entries["date"].dt.month == m
        )
        if not any(mask):
            return
        return self.entries.loc[mask]

    def get_selected_entry(self) -> pd.DataFrame | None:
        """Get selected entry row."""
        if self.selected_month is None or self.selected_entry is None:
            return
        rows = self.get_entries_month()
        if rows is None or rows.shape[0] <= self.selected_entry:
            return
        return rows.iloc[[self.selected_entry]]

    def set_selected_entry(self, i_entry: int):
        """Set selected entry."""
        self.selected_entry = i_entry
        row = self.get_selected_entry().iloc[0]
        if row is not None:
            self.new_type = None if row.isna()["type"] else row["type"]
            d = row["date"]
            if row.isna()["start"]:
                self.new_t_start = None
            else:
                start: pd.Timestamp = row["start"]
                start.replace(year=d.year, month=d.month, day=d.day)
                self.new_t_start = start.to_pydatetime()
            if row.isna()["end"]:
                self.new_t_end = None
            else:
                t_end: pd.Timestamp = row["end"]
                t_end.replace(year=d.year, month=d.month, day=d.day)
                self.new_t_end = t_end.to_pydatetime()

    async def async_update(self):
        """WorkClock states class."""
        t_perf = perf_counter()
        if self.reload:
            await self.async_update_states()
            await self.async_update_entries()
            self.reload = False
        if self.selected_month is None:
            self.selected_month = datetime.now()
        _LOGGER.debug("Update took %f", perf_counter() - t_perf)

    async def async_update_states(self) -> None:
        """Retrieve state from database."""
        states = await self.async_read_states()
        # finish when empty
        if not states.shape[0]:
            self.states = STATES_SCHEMA
            self.is_on = None
            self.states_day = STATES_DAY_SCHEMA
            return
        # no changes
        if states.equals(self.states):
            # finish if no changes and not on
            if not states.iloc[0]["state"]:
                return
            # calculate new end time and duration
            self.states_day.iloc[-1, self.states_day.columns == "end"] = utc_now_m()
            self.states_day["duration"] = (
                self.states_day["end"] - self.states_day["start"]
            ).astype(STATES_DAY_SCHEMA["duration"].dtype)
            return
        # write to states
        self.states = states
        self.is_on = self.states["state"].iloc[0] == 1
        # sort
        states_day = self.states.sort_values(by=["datetime"])
        # append last end
        if states_day.iloc[-1]["state"] == 1:
            states_day = pd.concat(
                (
                    states_day,
                    pd.DataFrame({"datetime": [utc_now_m()], "state": [False]}),
                ),
                ignore_index=True,
            )
        # drop duplicate states
        states_day = states_day.loc[states_day["state"].diff() != 0]
        # distribute times
        states_day.loc[states_day["state"] == 1, "start"] = states_day.loc[
            states_day["state"] == 1, "datetime"
        ].astype(STATES_DAY_SCHEMA["start"].dtype)
        states_day.loc[states_day["state"] == 0, "end"] = states_day.loc[
            states_day["state"] == 0, "datetime"
        ].astype(STATES_DAY_SCHEMA["end"].dtype)
        states_day["start"] = states_day["start"].ffill()
        states_day = states_day.loc[states_day["state"] == 0]
        # add date
        states_day["date"] = (
            states_day["start"]
            .dt.tz_convert(self.tz)
            .dt.date.astype(STATES_DAY_SCHEMA["date"].dtype)
        )
        # duration in minutes
        states_day["duration"] = (states_day["end"] - states_day["start"]).astype(
            STATES_DAY_SCHEMA["duration"].dtype
        )
        # grouping per day
        states_day = (
            states_day.groupby("date", group_keys=False)
            .agg({"duration": "sum", "start": "min", "end": "max"})
            .reset_index()
        )
        self.states_day = states_day[STATES_DAY_SCHEMA.columns]

    async def async_read_states(self) -> pd.DataFrame:
        """Retrieve state from database."""
        # read from domain table
        query = f"SELECT * FROM {TABLE_STATES} WHERE entity_id = '{self.entity_id}' ORDER BY datetime DESC;"  # noqa: S608
        result: Result | None = await self._async_exec_query(query)
        # finish when error
        if result is None:
            _LOGGER.warning("Error reading states for %s", self.entity_id)
            return STATES_SCHEMA
        # convert to dataframe
        states = pd.DataFrame(result.mappings().all())
        # finish when empty
        if not states.shape[0]:
            return STATES_SCHEMA
        # drop entity id
        states.drop(columns=["entity_id"], inplace=True)
        # fix dtypes
        states["datetime"] = states["datetime"].dt.tz_localize("UTC")
        for key in STATES_SCHEMA.columns:
            if key not in states.columns:
                states[key] = None
            states[key] = states[key].astype(STATES_SCHEMA[key].dtype)
        # drop duplicates
        return states.drop_duplicates()

    async def async_update_entries(self):
        """Update entries."""
        today = datetime.today()
        if today.month < 11:
            end_date = date(today.year + 1, 1, 1)
        else:
            end_date = date(today.year + 2, 1, 1)
        # read db
        self.entries = await self.async_read_entries()
        # create all dates
        entries = pd.DataFrame(
            {
                "date": pd.date_range(self.start_date, end_date).astype(
                    self.entries["date"].dtype
                ),
            }
        )
        # set calculated
        entries["calculated"] = True
        # concat datas
        self.entries = pd.concat(
            (self.entries, entries.loc[~entries["date"].isin(self.entries["date"])]),
            ignore_index=True,
        ).sort_values(by=["date", "start"])
        # add source staes to lines
        self.entries = self.entries.set_index("date", drop=False)
        # localize
        src = self.states_day.loc[self.states_day["date"] >= self.start_date].set_index(
            "date"
        )
        self.entries.loc[src.index, "src"] = src["duration"].dt.total_seconds() / 3600
        self.entries.loc[src.index, "src_start"] = (
            src["start"].dt.tz_convert(self.tz).dt.tz_localize(None)
        )
        self.entries.loc[src.index, "src_end"] = (
            src["end"].dt.tz_convert(self.tz).dt.tz_localize(None)
        )
        self.entries.index.name = "index"
        # calc entries
        self.calc_entries()

    async def async_read_entries(self) -> pd.DataFrame:
        """Read entries from db."""
        # read db
        query = f"SELECT * FROM {TABLE_ENTRIES} WHERE entity_id='{self.entity_id}' ORDER BY date ASC;"  # noqa: S608
        result: Result | None = await self._async_exec_query(query)
        if result is None:
            return ENTRIES_SCHEMA
        # convert to dataframe
        entries = pd.DataFrame(result.mappings().all())
        if not entries.shape[0]:
            return ENTRIES_SCHEMA
        entries.drop(columns=["entity_id"], inplace=True)
        # cat time and date
        for key in ["start", "end"]:
            mask = ~entries[key].isna()
            t = entries.loc[mask, key]
            entries[key] = pd.NaT
            entries[key] = entries[key].astype(ENTRIES_SCHEMA[key].dtype)
            entries.loc[mask, key] = (
                entries.loc[mask, "date"].astype("datetime64[s]") + t
            )
        # fix dtypes
        for key in ENTRIES_SCHEMA.columns:
            if key not in entries.columns:
                entries[key] = None
            entries[key] = entries[key].astype(ENTRIES_SCHEMA[key].dtype)
        # set calculated
        entries["calculated"] = False
        return pd.concat((ENTRIES_SCHEMA, entries))

    def calc_entries(self):
        """Calculate entries values."""
        # write public holidays
        self.public_hol()

        mask_hol = self.entries["holiday"] != ""
        mask_weekend = self.entries["date"].dt.dayofweek >= 5
        mask_work_day = ~mask_hol & ~mask_weekend
        mask_work = self.entries["type"].isin(ENTRY_TYPES_WORK)
        mask_off = self.entries["type"].isin(ENTRY_TYPES_OFF)
        mask_today = self.entries["date"] <= datetime.now(tz=self.tz).replace(
            tzinfo=None
        )

        # target time
        self.entries.loc[mask_work_day, "tar_hours"] = 7
        self.entries.loc[mask_work & mask_hol & ~mask_weekend, "tar_hours"] = 7
        # Heiligabend, Silvester
        mask = (
            mask_work
            & (self.entries["date"].dt.month == 12)
            & self.entries["date"].dt.day.isin([24, 31])
        )
        self.entries.loc[mask, "tar_hours"] = 4.75

        # fill empty types
        self.entries.loc[
            mask_work_day & mask_today & ~self.entries["type"].isin(ENTRY_TYPES), "type"
        ] = "KG"
        mask_work = self.entries["type"].isin(ENTRY_TYPES_WORK)

        # fill work off
        self.entries.loc[mask_off, "time"] = timedelta(seconds=0)
        self.entries.loc[mask_off, "time_sum"] = 0.0
        self.entries.loc[mask_off, "time_booked"] = 7.0
        self.entries.loc[self.entries["type"] == "FG", "time_booked"] = 0.0

        # mask for calculation
        mask_calc = self.entries["type"].isin(["KG", "MA"]) & (
            self.entries["start"].isna() | self.entries["end"].isna()
        )
        mask_clear = self.entries["date"].diff(-1) == pd.Timedelta(0, "s")

        i_st = self.entries.index[0]
        while i_st <= self.entries.index[-1]:
            # find ranges to calculate
            mask = mask_calc & (self.entries.index >= i_st)
            if mask.any():
                i_end = mask.idxmax()
            else:
                i_end = self.entries.index[-1] + pd.Timedelta(1, "d")
            mask = (
                mask_work & (self.entries.index >= i_st) & (self.entries.index < i_end)
            )

            # calculate time sum for entered lines
            if any(mask):
                # calculate break
                self.entries.loc[mask, "break"] = self.entries.loc[mask].apply(
                    lambda line: calc_breaks(line["start"], line["end"]),
                    axis=1,
                )
                # calculate time
                self.entries.loc[mask, "time"] = (
                    self.entries.loc[mask, "end"]
                    - self.entries.loc[mask, "start"]
                    - self.entries.loc[mask, "break"]
                ).astype(ENTRIES_SCHEMA["time"].dtype)
                grps = self.entries.loc[mask, ["date", "time"]].groupby("date").sum()
                grps.index = grps.index.date
                self.entries.loc[grps.index, "time_sum"] = (
                    grps["time"].dt.total_seconds() / 3600
                ).clip(0, 11.0)
                self.entries.loc[mask_clear, "time_sum"] = np.nan
                self.entries.loc[mask, "time_booked"] = self.entries.loc[
                    mask, "time_sum"
                ]

            # calc times
            self.entries.loc[
                mask_clear, ["time_sum", "time_booked", "src", "tar_hours"]
            ] = None
            if i_end in self.entries.index:
                self.calc_times(i_end)

            i_st = i_end + pd.Timedelta(1, "d")

        self.entries.loc[
            mask_clear, ["time_sum", "time_booked", "src", "tar_hours"]
        ] = None

        # overtime
        self.entries.loc[~mask_clear, "overhours"] = (
            self.entries.loc[~mask_clear, "time_booked"]
            - self.entries.loc[~mask_clear, "tar_hours"]
        )
        # time_account
        mask = self.entries["type"].isin(ENTRY_TYPES)
        time_account = (
            self.entries.loc[:, ["src", "time_sum"]]
            .replace(np.nan, 0.0)
            .diff(-1, axis=1)["src"]
        )
        time_account = time_account.cumsum() + self.start_fz
        self.entries.loc[mask, "time_acc"] = time_account.loc[mask]
        # active
        mask = self.entries["type"].isin(ENTRY_TYPES_WORK) & (
            self.entries["type"] != "DM"
        )
        grps = self.entries.loc[mask, ["date", "time"]].groupby("date").sum()
        grps.index = grps.index.date
        self.entries.loc[grps.index, "aktiv"] = grps["time"]
        # t_rest
        self.entries.loc[:, "rest"] = (
            self.entries["start"] - self.entries["end"].shift(1).ffill()
        ).astype(ENTRIES_SCHEMA["rest"].dtype)

    def calc_times(self, i):
        """Calculate times for lines."""
        row = self.entries.loc[[i]]
        if row.shape[0] != 1:
            return
        row = self.calc_time(row.iloc[0], self.calc_fz(i))
        if row is None:
            return
        self.entries.loc[i] = row

    def calc_time(self, row: pd.Series, fz: pd.Timedelta) -> pd.Series | None:
        """Calculate times for lines."""
        if not pd.isna(row["start"]) and not pd.isna(row["end"]):
            return
        if row["type"] not in ["KG", "MA"]:
            return
        # src
        if pd.isna(row["src"]):
            dur_m = pd.Timedelta(0, "h")
        else:
            dur_m = pd.Timedelta(row["src"], "h")
        dur_m = dur_m.round("min")
        # init
        start_min = row["date"].replace(hour=6, minute=30, second=0)
        end_max = row["date"].replace(hour=19, minute=0, second=0)
        if pd.isna(row["start"]) and pd.isna(row["end"]):
            if pd.isna(row["src_end"]):
                return
            row["end"] = row["src_end"]
        if not pd.isna(row["start"]) and pd.isna(row["end"]):
            row["start"] = max(row["start"], start_min)
        elif pd.isna(row["start"]) and not pd.isna(row["end"]):
            row["end"] = min(row["end"], end_max)
        dur_max = pd.Timedelta(10, "h")
        tar_m = max(pd.Timedelta(0, "m"), min(dur_max, dur_m + fz))
        # calc t start
        calc_t_start = pd.isna(row["start"])
        for m in range(120):
            m_d = tar_m + pd.Timedelta(m, "m")
            if calc_t_start:
                row["start"] = max(row["end"] - m_d, start_min)
            else:
                row["end"] = min(row["start"] + m_d, end_max)
            row["break"] = calc_breaks(row["start"], row["end"])
            row["time"] = row["end"] - row["start"] - row["break"]
            if abs(row["time"] - tar_m) < pd.Timedelta(0.9, "m"):
                break
            if calc_t_start:
                if row["start"] <= start_min:
                    break
            elif row["end"] >= end_max:
                break
        else:
            _LOGGER.error(
                "%s could not be calculated for %s",
                "t_start" if calc_t_start else "t_end",
                row["date"],
            )
        row["time_sum"] = np.clip(row["time"].total_seconds() / 3600, 0, 11.0)
        row["time_booked"] = row["time_sum"]
        return row

    def calc_fz(self, i) -> pd.Timedelta:
        """Calculate FZ for date."""
        mask = self.entries.index < i
        time_account = (
            self.entries.loc[mask, ["src", "time_sum"]]
            .replace(np.nan, 0.0)
            .diff(-1, axis=1)["src"]
        )
        dt = time_account.sum() + self.start_fz
        return pd.Timedelta(round(dt * 60), "m")

    def public_hol(self):
        """Calculate public holidays."""
        # init
        self.entries["holiday"] = ""
        days = self.entries["date"]
        years = days.dt.year.unique()

        # Easter sunday
        a = years % 19
        b = years % 4
        c = years % 7
        k = (years / 100).astype(int)
        p = ((13 + 8 * k) / 25).astype(int)
        q = (k / 4).astype(int)
        m = (15 - p + k - q) % 30
        n = (4 + k - q) % 7
        d = (19 * a + m) % 30
        e = (2 * b + 4 * c + 6 * d + n) % 7
        td = ((21 + d + e) * 24 * 3600).astype("timedelta64[s]")
        es = pd.Series([datetime(Y, 3, 1) for Y in years]) + td

        # new years
        ds = pd.Series([datetime(Y, 1, 1) for Y in years])
        self.entries.loc[self.entries["date"].isin(ds), "holiday"] = "Neujahr"
        # Dreikönig
        ds = pd.Series([datetime(Y, 1, 6) for Y in years])
        self.entries.loc[self.entries["date"].isin(ds), "holiday"] = "Dreikönig"
        # Karfreitag
        ds = es - timedelta(days=2)
        self.entries.loc[self.entries["date"].isin(ds), "holiday"] = "Karfreitag"
        # Ostersonntag
        ds = es
        self.entries.loc[self.entries["date"].isin(ds), "holiday"] = "Ostersonntag"
        # Ostermontag
        ds = es + timedelta(days=1)
        self.entries.loc[self.entries["date"].isin(ds), "holiday"] = "Ostermontag"
        # Maifeiertag
        ds = pd.Series([datetime(Y, 5, 1) for Y in years])
        self.entries.loc[self.entries["date"].isin(ds), "holiday"] = "Maifeiertag"
        # Christi Himmelfahrt
        ds = es + timedelta(days=39)
        self.entries.loc[
            self.entries["date"].isin(ds), "holiday"
        ] = "Christi Himmelfahrt"
        # Pfingstmontag
        ds = es + timedelta(days=50)
        self.entries.loc[self.entries["date"].isin(ds), "holiday"] = "Pfingstmontag"
        # Fronleichnam
        ds = es + timedelta(days=60)
        self.entries.loc[self.entries["date"].isin(ds), "holiday"] = "Fronleichnam"
        # Mariä Himmelfahrt
        ds = pd.Series([datetime(Y, 8, 15) for Y in years])
        self.entries.loc[self.entries["date"].isin(ds), "holiday"] = "Mariä Himmelfahrt"
        # Tag der dt. Einheit
        ds = pd.Series([datetime(Y, 10, 3) for Y in years])
        self.entries.loc[
            self.entries["date"].isin(ds), "holiday"
        ] = "Tag der dt. Einheit"
        # Allerheiligen
        ds = pd.Series([datetime(Y, 11, 1) for Y in years])
        self.entries.loc[self.entries["date"].isin(ds), "holiday"] = "Allerheiligen"
        # 1. Weihnachtstag
        ds = pd.Series([datetime(Y, 12, 25) for Y in years])
        self.entries.loc[self.entries["date"].isin(ds), "holiday"] = "1. Weihnachtstag"
        # 2. Weihnachtstag
        ds = pd.Series([datetime(Y, 12, 26) for Y in years])
        self.entries.loc[self.entries["date"].isin(ds), "holiday"] = "2. Weihnachtstag"

    async def async_write_state(self, state: bool):
        """Add state to db."""
        # check if time exists
        dt = utc_now_m().replace(tzinfo=None)
        query = (
            f"SELECT * FROM {TABLE_STATES} "  # noqa: S608
            f"WHERE entity_id = '{self.entity_id}' "
            f"AND datetime = '{dt}' "
            f"AND state = '{int(state)}' "
            f"ORDER BY datetime DESC;"
        )
        result = await self._async_exec_query(query)
        if result is not None and result.all():
            _LOGGER.warning("%s already exists", dt.isoformat())
            return
        # add to db
        query = (
            f"INSERT INTO {TABLE_STATES} (datetime, entity_id, state) VALUES"
            f"('{dt}', '{self.entity_id}', '{int(state)}');"  # noqa: DTZ003
        )
        result = await self._async_exec_query(query)
        if result is None or result.rowcount < 1:
            _LOGGER.error("Could not add state %s", dt.isoformat())
            return
        self.reload = True

    async def async_delete_state(self) -> bool:
        """Remove state from states."""
        if self.selected_state is None:
            _LOGGER.error("No state selected")
            return False
        date_time = self.selected_state
        if date_time.tzinfo is None:
            date_time = date_time.replace(tzinfo=self.tz)
        date_time = date_time.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        query = f"DELETE FROM {TABLE_STATES} WHERE entity_id = '{self.entity_id}' AND datetime IN ('{date_time}', '{date_time.replace(tzinfo=None)}');"  # noqa: S608
        result = await self._async_exec_query(query)
        if result is None or result.rowcount <= 0:
            _LOGGER.error(
                "Could not delete state %s\n%s", self.selected_state.isoformat(), query
            )
            return False
        _LOGGER.info(
            "Deleted %d rows for state %s",
            result.rowcount,
            self.selected_state.isoformat(),
        )
        self.selected_state = None
        self.reload = True
        return True

    async def async_edit_state(self) -> bool:
        """Edit datetime of state in states."""
        new_time = self.new_state
        if new_time.tzinfo is None:
            new_time = new_time.replace(tzinfo=self.tz)
        new_time = new_time.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)

        selected_time = self.selected_state
        if selected_time.tzinfo is None:
            selected_time = selected_time.replace(tzinfo=self.tz)
        selected_time = selected_time.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)

        query = f"UPDATE {TABLE_STATES} SET datetime = '{new_time}' WHERE entity_id = '{self.entity_id}' AND datetime = '{selected_time}';"  # noqa: S608
        result = await self._async_exec_query(query)

        if result is None or result.rowcount < 1:
            _LOGGER.error("Could not edit state %s", self.selected_state.isoformat())
            return False
        _LOGGER.info(
            "Edited %d rows (%s to %s)",
            result.rowcount,
            selected_time.isoformat(),
            new_time.isoformat(),
        )

        self.selected_state = None
        self.reload = True

        return True

    def _debug_new_entry(self) -> pd.Series | None:
        # no entry selected
        row = self.get_selected_entry()
        if row is None:
            _LOGGER.error("No entry selected")
            return
        i = row.index[0]
        row = row.iloc[0]
        # debug entry types
        if self.new_type not in ENTRY_TYPES:
            _LOGGER.error("Unknown type %s", self.new_type)
            return
        # debug start end
        if self.new_type in ["D", "DM"] and (
            self.new_t_start is None or self.new_t_end is None
        ):
            _LOGGER.error(
                "Start and end time need to be set for line type %s", self.new_type
            )
            return
        # calc time
        if self.new_type in ENTRY_TYPES_OFF:
            self.new_t_start = None
            self.new_t_end = None
        elif self.new_t_start is None or self.new_t_end is None:
            new_row = self.get_selected_entry().iloc[0]
            new_row["type"] = self.new_type
            new_row["start"] = pd.NaT if self.new_t_start is None else self.new_t_start
            new_row["end"] = pd.NaT if self.new_t_end is None else self.new_t_end
            new_row = self.calc_time(new_row, self.calc_fz(i))
            self.new_t_start = new_row["start"]
            self.new_t_end = new_row["end"]
        return row

    async def async_add_entry(self) -> bool:
        """Add entry."""
        row = self._debug_new_entry()
        if row is None:
            return False
        # selected entry is not calculated
        if not row["calculated"]:
            _LOGGER.error("Entry is not alculated, call 'edit line' instead")
            return False

        # check if time exists
        query = f"SELECT * FROM {TABLE_ENTRIES} "  # noqa: S608
        query = f"{query} WHERE entity_id='{self.entity_id}'"
        query = f"{query} AND date='{row['date'].date()}'"
        query = f"{query} AND type='{self.new_type}'"
        if pd.isna(self.new_t_start):
            query = f"{query} AND start IS NULL"
        else:
            query = f"{query} AND start='{self.new_t_start.time()}'"
        if pd.isna(row["end"]):
            query = f"{query} AND end IS NULL"
        else:
            query = f"{query} AND end='{self.new_t_end.time()}'"
        query = f"{query};"
        result: Result | None = await self._async_exec_query(query)
        if result is None:
            _LOGGER.error("Error looking for existing row")
            return False
        if result.all():
            _LOGGER.error("Error new entry already exists")
            return False

        # add to db
        query = f"INSERT INTO {TABLE_ENTRIES}"
        query = f"{query} (entity_id, date, type, start, end)"
        query = f"{query} VALUES('{self.entity_id}','{row['date'].date()}','{self.new_type}'"
        if self.new_type in ENTRY_TYPES_OFF:
            query = f"{query},NULL,NULL"
        else:
            query = f"{query},'{self.new_t_start.time()}','{self.new_t_end.time()}'"
        query = f"{query});"

        result: Result | None = await self._async_exec_query(query)
        if result is None or result.rowcount < 1:
            _LOGGER.warning("Could not add new entry to db")
            return False

        self.reload = True
        return True

    async def async_edit_entry(self) -> bool:
        """Edit entry."""
        row = self._debug_new_entry()
        if row is None:
            return False
        # calculated line
        if row["calculated"]:
            _LOGGER.error("Entry is calculated, call 'add line' instead")
            return False
        # test for change
        if (
            (row["type"] == self.new_type)
            and (row["start"] == self.new_t_start)
            and (row["end"] == self.new_t_end)
        ):
            _LOGGER.error("No change in line data")
            return False
        # update db
        query = f"UPDATE {TABLE_ENTRIES} "
        query = f"{query} SET type='{self.new_type}'"
        if self.new_type in ENTRY_TYPES_OFF:
            query = f"{query}, start=NULL, end=NULL"
        else:
            query = f"{query}, start='{self.new_t_start.time()}'"
            query = f"{query}, end='{self.new_t_end.time()}'"
        query = f"{query} WHERE entity_id='{self.entity_id}'"
        query = f"{query} AND date='{row['date'].date()}'"
        query = f"{query} AND type='{row['type']}'"
        if pd.isna(row["start"]):
            query = f"{query} AND start IS NULL"
        else:
            query = f"{query} AND start='{row['start'].time()}'"
        if pd.isna(row["end"]):
            query = f"{query} AND end IS NULL"
        else:
            query = f"{query} AND end='{row['end'].time()}'"
        query = f"{query};"
        result: Result | None = await self._async_exec_query(query)
        if result is None or result.rowcount < 1:
            _LOGGER.warning("Could not edit line in db")
            return False
        self.reload = True
        return True

    async def async_delete_entry(self) -> bool:
        """Delete entry from db."""
        row = self.get_selected_entry()
        if row is None:
            _LOGGER.error("No entry selected")
            return False
        row = row.iloc[0]

        query = f"DELETE FROM {TABLE_ENTRIES} "  # noqa: S608
        query = f"{query} WHERE entity_id='{self.entity_id}'"
        query = f"{query} AND date='{row['date'].date()}'"
        query = f"{query} AND type='{row['type']}'"
        if pd.isna(self.new_t_start):
            query = f"{query} AND start IS NULL"
        else:
            query = f"{query} AND start='{row['start'].time()}'"
        if pd.isna(row["end"]):
            query = f"{query} AND end IS NULL"
        else:
            query = f"{query} AND end='{row['end'].time()}'"
        query = f"{query};"
        result: Result | None = await self._async_exec_query(query)
        if result is None or result.rowcount < 1:
            _LOGGER.error("Error deleting entry")
            return False

        self.reload = True
        return True


def calc_breaks(t_start: datetime, t_end: datetime) -> pd.Timedelta:
    """Calculate breaks."""
    return calc_break(t_start, t_end, False) + calc_break(t_start, t_end, True)


def calc_break(t_start: datetime, t_end: datetime, noon_break: bool) -> pd.Timedelta:
    """Calculate break."""
    if pd.isna(t_start) or pd.isna(t_end) or t_end <= t_start:
        return pd.Timedelta(0, "s")
    # standard start and end times
    if noon_break:
        if t_start.weekday() >= 4:
            pause_st = pd.Timestamp(t_start).replace(hour=12, minute=15, second=0)
        else:
            pause_st = pd.Timestamp(t_start).replace(hour=12, minute=0, second=0)
        pause_end = pd.Timestamp(t_start).replace(hour=12, minute=45, second=0)
    else:
        pause_st = pd.Timestamp(t_start).replace(hour=9, minute=0, second=0)
        pause_end = pd.Timestamp(t_start).replace(hour=9, minute=15, second=0)
    tmp_pause_st = max(min(t_start, pause_end), pause_st)
    tmp_pause_end = min(max(t_end, pause_st), pause_end)
    return tmp_pause_end - tmp_pause_st


async def async_get_sessionmaker(
    hass: HomeAssistant, db_url: str
) -> [
    scoped_session | None,
    bool,
]:
    """Ger sessionmaker from db_url."""
    instance = get_instance(hass)
    sessmaker: scoped_session | None
    sql_data = _async_get_or_init_domain_data(hass)
    uses_recorder_db = db_url == instance.db_url
    use_database_executor = False

    if uses_recorder_db and instance.dialect_name == SupportedDialect.SQLITE:
        use_database_executor = True
        assert instance.engine is not None
        sessmaker = scoped_session(sessionmaker(bind=instance.engine, future=True))
        await get_instance(hass).async_add_executor_job(
            create_tables, sessmaker, instance.engine
        )
    # For other databases we need to create a new engine since
    # we want the connection to use the default timezone and these
    # database engines will use QueuePool as its only sqlite that
    # needs our custom pool. If there is already a session maker
    # for this db_url we can use that so we do not create a new engine
    # for every sensor.
    elif db_url in sql_data.session_makers_by_db_url:
        sessmaker = sql_data.session_makers_by_db_url[db_url]
        await hass.async_add_executor_job(
            create_tables,
            sessmaker,
            sqlalchemy.create_engine(db_url, future=True),
        )
    elif sessmaker := await hass.async_add_executor_job(
        _validate_and_get_session_maker_for_db_url, db_url
    ):
        sql_data.session_makers_by_db_url[db_url] = sessmaker
        await hass.async_add_executor_job(
            create_tables,
            sessmaker,
            sqlalchemy.create_engine(db_url, future=True),
        )
    else:
        return None, use_database_executor
    return sessmaker, use_database_executor


def create_table(
    sess: scoped_session,
    engine: Engine,
    table_name: str,
    columns: list[sqlalchemy.Column],
):
    """Create table in database."""
    metadata = sqlalchemy.MetaData()
    # Create a table with the appropriate Columns
    try:
        sess.execute(text(f"DESCRIBE {table_name};"))
    except SQLAlchemyError:
        sqlalchemy.Table(table_name, metadata, *columns)
        # Implement the creation
        metadata.create_all(bind=engine)


def create_tables(sessmaker: scoped_session, engine: Engine):
    """Create Tables for states and entries."""
    # create table for states
    create_table(
        sessmaker,
        engine,
        TABLE_STATES,
        [
            sqlalchemy.Column(
                "datetime", sqlalchemy.DateTime, primary_key=True, nullable=False
            ),
            sqlalchemy.Column("entity_id", sqlalchemy.String(128)),
            sqlalchemy.Column("state", sqlalchemy.Boolean),
        ],
    )
    # create table for entries
    create_table(
        sessmaker,
        engine,
        TABLE_ENTRIES,
        [
            sqlalchemy.Column("date", sqlalchemy.Date, nullable=False),
            sqlalchemy.Column("entity_id", sqlalchemy.String(128)),
            sqlalchemy.Column("type", sqlalchemy.String(2)),
            sqlalchemy.Column("start", sqlalchemy.Time),
            sqlalchemy.Column("end", sqlalchemy.Time),
        ],
    )


async def async_delete_state(
    hass: HomeAssistant,
    sessmaker: scoped_session,
    use_db_executor: bool,
    table: str,
    state: dict,
) -> int:
    """Delete state from database."""
    query = f"DELETE FROM {table} WHERE"  # noqa: S608
    for key, v in state.items():
        if v is None:
            query = f"{query} {key} IS NULL AND"
        elif key not in [
            "context_id_bin",
            "context_user_id_bin",
            "context_parent_id_bin",
        ]:
            query = f"{query} {key} = '{v}' AND"
    if fnmatch.fnmatch(query, "* AND"):
        query = query[:-4]
    query = f"{query};"
    result = await async_exec_query(hass, sessmaker, use_db_executor, query)
    if result is None:
        return 0
    return result.rowcount


async def async_edit_state(
    hass: HomeAssistant,
    sessmaker: scoped_session,
    use_db_executor: bool,
    table: str,
    state: dict,
    new_state: dict,
) -> int:
    """Edit state in database."""
    query = f"UPDATE {table} SET"
    for key, v in new_state.items():
        if v is None:
            query = f"{query} {key} = NULL,"
        else:
            query = f"{query} {key} = '{v}',"
    if fnmatch.fnmatch(query, "*,"):
        query = query[:-1]
    query = f"{query} WHERE"
    for key, v in state.items():
        if v is None:
            query = f"{query} {key} IS NULL AND"
        else:
            query = f"{query} {key} = '{v}' AND"
    if fnmatch.fnmatch(query, "* AND"):
        query = query[:-4]
    query = f"{query};"
    result = await async_exec_query(hass, sessmaker, use_db_executor, query)
    if result is None:
        return 0
    return result.rowcount


async def async_exec_query(
    hass: HomeAssistant, sessmaker: scoped_session, use_db_executor: bool, query: str
) -> [Result | None]:
    """Execute query in database."""
    if use_db_executor:
        return await get_instance(hass).async_add_executor_job(
            _exec_query, sessmaker, query
        )
    return await hass.async_add_executor_job(_exec_query, sessmaker, query)


def _exec_query(sessmaker: scoped_session, query: str) -> [Result | None]:
    """Execute query in database."""
    sess: scoped_session = sessmaker()
    try:
        lambda_stmt = _generate_lambda_stmt(query)
        result: Result = sess.execute(lambda_stmt)
    except SQLAlchemyError as err:
        _LOGGER.error(
            "Error executing query %s: %s",
            query,
            redact_credentials(str(err)),
        )
        sess.rollback()
        sess.close()
        return
    sess.commit()
    sess.close()
    return result
