"""Work Clock Input datetime."""
from __future__ import annotations

import datetime as py_datetime
import logging
from zoneinfo import ZoneInfo

import voluptuous as vol

from homeassistant.components.input_datetime import (
    CONF_HAS_DATE,
    CONF_HAS_TIME,
    InputDatetime,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME, CONF_TIME_ZONE
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import WorkClockDataUpdateCoordinator
from .const import DOMAIN
from .entity import WorkClockEntity

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Initialize WorkClock config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]
    entities = [WorkClockNewDatetimeState(coordinator, entry)]
    async_add_entities(entities)


class WorkClockNewDatetimeState(WorkClockEntity, InputDatetime):
    """Representation of work clock input datetime."""

    def __init__(
        self,
        coordinator: WorkClockDataUpdateCoordinator,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the WorkClock input datetime."""
        super().__init__(coordinator, config_entry)
        self.editable = True
        self._config = {
            CONF_HAS_DATE: True,
            CONF_HAS_TIME: True,
        }
        self._current_datetime = None
        self.tz = ZoneInfo(config_entry.options.get(CONF_TIME_ZONE))
        _LOGGER.info("Added %s", self.entity_id)

    async def async_set_value(self, value: py_datetime) -> None:
        """Change the date/time."""
        self.async_set_datetime(datetime=value)
        await self.coordinator.async_request_refresh()

    @callback
    def async_set_datetime(self, date=None, time=None, datetime=None, timestamp=None):
        """Set a new date / time."""
        if timestamp:
            datetime = py_datetime.fromtimestamp(timestamp, tz=self.tz)

        if datetime:
            datetime = datetime.astimezone(self.tz)
            date = datetime.date()
            time = datetime.time()

        if not self.has_date:
            date = None

        if not self.has_time:
            time = None

        if not date and not time:
            raise vol.Invalid("Nothing to set")

        if not date:
            date = self._current_datetime.date()

        if not time:
            time = self._current_datetime.time()

        self.coordinator.client.new_state = py_datetime.datetime.combine(
            date, time, self.tz
        )

    @property
    def name(self):
        """Return the name of the input datetime."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"New {name.capitalize()} State"

    @property
    def icon(self):
        """Return the icon of the input datetime."""
        return "mdi:timer-edit-outline"

    @property
    def state(self):
        """Return the state of the component."""
        self._current_datetime = self.coordinator.client.new_state
        return super().state

    @property
    def extra_state_attributes(self):
        """Return the state attributes."""
        my_attrs = {}
        for key in ("selected_state", "new_state"):
            val = getattr(self.coordinator.client, key)
            my_attrs[key] = val
        attrs = super().extra_state_attributes
        if attrs is None:
            return my_attrs
        attrs.update(my_attrs)
        return attrs
