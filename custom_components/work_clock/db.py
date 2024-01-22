"""Database functions for work clock."""
from datetime import date, datetime, timedelta
import fnmatch
import logging
from time import perf_counter
from zoneinfo import ZoneInfo

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
    ENTRIES_SCHEMA,
    ENTRY_TYPES_OFF,
    STATES_DAY_SCHEMA,
    STATES_SCHEMA,
    TABLE_ENTRIES,
    TABLE_STATES,
)
from .util import utc_now_m

_LOGGER = logging.getLogger(__name__)


class WorkClockDbClient:
    """WorkClock states class."""

    states: pd.DataFrame = STATES_SCHEMA
    states_day: pd.DataFrame = STATES_DAY_SCHEMA
    entries: pd.DataFrame = ENTRIES_SCHEMA
    is_on: bool | None = None
    selected_state: datetime | None = None
    new_state: datetime | None = None
    selected_month: datetime | None = None
    selected_entry: int | None = None

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
        self.hass: HomeAssistant = hass
        self.sessionmaker: scoped_session = sessmaker
        self._use_database_executor: bool = use_database_executor

    async def _async_exec_query(self, query: str) -> Result | None:
        return await async_exec_query(
            self.hass, self.sessionmaker, self._use_database_executor, query
        )

    async def async_update(self):
        """WorkClock states class."""
        t_perf = perf_counter()
        await self.async_update_states()
        await self.async_update_entries()
        if self.selected_month is None:
            self.selected_month = datetime.now()
        _LOGGER.info("Update took %f", perf_counter() - t_perf)

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
        for key in STATES_SCHEMA.columns:
            if key not in states.columns:
                states[key] = None
            states[key] = states[key].astype(STATES_SCHEMA[key].dtype)
        # add tz
        states["datetime"] = states["datetime"].dt.tz_convert(self.tz)
        # self.states["datetime"] = self.states["datetime"].dt.round("60s")
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
        self.entries.loc[src.index, "src_start"] = src["start"].astype(
            self.entries["src_start"].dtype
        )
        self.entries.loc[src.index, "src_end"] = src["end"].astype(
            self.entries["src_end"].dtype
        )
        # write public holidays
        self.public_hol()

        # mask for work days
        mask_work = (self.entries["holiday"] == "") & (
            self.entries["date"].dt.dayofweek < 5
        )
        # mask for items before today
        mask_today = self.entries["date"] <= datetime.now(tz=self.tz).replace(
            tzinfo=None
        )

        # target
        self.entries.loc[mask_work, "tar_hours"] = 7
        self.entries.loc[
            ~self.entries["type"].isna() & (self.entries["holiday"] != ""), "tar_hours"
        ] = 7
        self.entries.loc[
            ~self.entries["type"].isna() & (self.entries["date"].dt.dayofweek >= 5),
            "tar_hours",
        ] = 0
        # Heiligabend, Silvester
        mask = (
            mask_work
            & (self.entries["date"].dt.month == 12)
            & self.entries["date"].dt.day.isin([24, 31])
        )
        self.entries.loc[mask, "tar_hours"] = 4.75

        # fill empty types
        mask = mask_work & mask_today & self.entries["type"].isna()
        self.entries.loc[mask, "type"] = "KG"

        # fill work off
        mask_off = self.entries["type"].isin(ENTRY_TYPES_OFF)
        self.entries.loc[mask_off, "time"] = timedelta(seconds=0)
        self.entries.loc[mask_off, "time_sum"] = 0.0
        self.entries.loc[mask_off, "time_booked"] = 7.0
        self.entries.loc[self.entries["type"] == "FG", "time_booked"] = 0.0

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
        entries["start"] = entries["date"] + " " + entries["start"]
        entries["end"] = entries["date"] + " " + entries["end"]
        # fix dtypes
        for key in ENTRIES_SCHEMA.columns:
            if key not in entries.columns:
                entries[key] = None
            entries[key] = entries[key].astype(ENTRIES_SCHEMA[key].dtype)
        # set calculated
        entries["calculated"] = False
        _LOGGER.info(entries)
        return pd.concat((ENTRIES_SCHEMA, entries))

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
        dt = utc_now_m()
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

    async def async_delete_state(self) -> bool:
        """Remove state from states."""
        if self.selected_state is None:
            _LOGGER.error("No state selected")
            return False
        date_time = self.selected_state
        if date_time.tzinfo is None:
            date_time = date_time.replace(tzinfo=self.tz)
        date_time = date_time.astimezone(ZoneInfo("UTC"))
        query = f"DELETE FROM {TABLE_STATES} WHERE entity_id = '{self.entity_id}' AND datetime = '{date_time}';"  # noqa: S608
        result = await self._async_exec_query(query)
        if result is None:
            _LOGGER.error("Could not delete state %s", self.selected_state.isoformat)
            return False
        _LOGGER.info(
            "Deleted %d rows for state %s",
            result.rowcount,
            self.selected_state.isoformat,
        )
        return result.rowcount > 0

    async def async_edit_state(self) -> bool:
        """Edit datetime of state in states."""
        new_time = self.new_state
        if new_time.tzinfo is None:
            new_time = new_time.replace(tzinfo=self.tz)
        new_time = new_time.astimezone(ZoneInfo("UTC"))

        selected_time = self.selected_state
        if selected_time.tzinfo is None:
            selected_time = selected_time.replace(tzinfo=self.tz)
        selected_time = selected_time.astimezone(ZoneInfo("UTC"))

        query = f"UPDATE {TABLE_STATES} SET datetime = '{new_time}' WHERE entity_id = '{self.entity_id}' AND datetime = '{selected_time}';"  # noqa: S608
        result = await self._async_exec_query(query)
        if result is None:
            return False
        _LOGGER.info(
            "Edited %d rows (%s to %s)",
            result.rowcount,
            selected_time.isoformat,
            new_time.isoformat,
        )
        return result.rowcount == 1


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
