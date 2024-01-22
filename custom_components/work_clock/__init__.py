"""The work_clock component."""
from __future__ import annotations

from asyncio import timeout as async_timeout
from datetime import timedelta
import logging

import voluptuous as vol

from homeassistant.components.recorder import CONF_DB_URL
from homeassistant.components.sql.util import resolve_db_url
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME, CONF_TIME_ZONE
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.debounce import Debouncer
from homeassistant.helpers.typing import ConfigType
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .const import (
    CONF_START_DATE,
    CONF_START_FG,
    CONF_START_FZ,
    CONF_WORKHOURS,
    DOMAIN,
    PLATFORMS,
    STARTUP_MESSAGE,
)
from .db import WorkClockDbClient, async_get_sessionmaker

_LOGGER = logging.getLogger(__name__)


ITEM_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_NAME): cv.string,
        vol.Optional(CONF_DB_URL): cv.string,
        vol.Required(CONF_TIME_ZONE): cv.string,
        vol.Required(CONF_WORKHOURS): cv.positive_int,
        vol.Required(CONF_START_DATE): cv.date,
        vol.Required(CONF_START_FG): cv.small_float,
        vol.Required(CONF_START_FZ): cv.small_float,
    }
)

CONFIG_SCHEMA = vol.Schema(
    {vol.Optional(DOMAIN): vol.All(cv.ensure_list, [ITEM_SCHEMA])},
    extra=vol.ALLOW_EXTRA,
)


async def async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Update listener for options."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up this integration using YAML is not supported."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up WorkClock from a config entry."""
    if hass.data.get(DOMAIN) is None:
        hass.data.setdefault(DOMAIN, {})
        _LOGGER.info(STARTUP_MESSAGE)

    sessmaker, use_database_executor = await async_get_sessionmaker(
        hass, resolve_db_url(hass, entry.options.get(CONF_DB_URL))
    )
    if sessmaker is None:
        return False
    client: WorkClockDbClient = WorkClockDbClient(
        hass, entry.options, sessmaker, use_database_executor
    )
    coordinator = WorkClockDataUpdateCoordinator(hass, client=client)
    await coordinator.async_refresh()

    if not coordinator.last_update_success:
        raise ConfigEntryNotReady

    hass.data[DOMAIN][entry.entry_id] = coordinator

    entry.async_on_unload(entry.add_update_listener(async_update_listener))

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


class WorkClockDataUpdateCoordinator(DataUpdateCoordinator):
    """WorkClock coordinator."""

    def __init__(self, hass: HomeAssistant, client: WorkClockDbClient) -> None:
        """Initialize WorkClock coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            # Name of the data. For logging purposes.
            name="WorkClock",
            # Polling interval. Will only be polled if there are subscribers.
            update_interval=timedelta(seconds=300),
            request_refresh_debouncer=Debouncer(
                hass,
                _LOGGER,
                cooldown=1,
                immediate=True,
                function=self.async_refresh,
            ),
        )
        self.client: WorkClockDbClient = client

    async def _async_update_data(self):
        """Fetch data from Databse."""
        async with async_timeout(10):
            return await self.client.async_update()


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload WorkClock config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
