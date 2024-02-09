"""Work Clock Buttons."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import logging
from typing import Any

import numpy as np
import pandas as pd

from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import WorkClockDataUpdateCoordinator
from .const import DATE_FORMAT, DOMAIN, MONTH_FORMAT
from .entity import WorkClockEntity

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Initialize WorkClock config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]
    entities = [
        WorkClockButtonReload(coordinator, entry),
        WorkClockButtonDeleteState(coordinator, entry),
        WorkClockButtonEditState(coordinator, entry),
        WorkClockButtonAddEntry(coordinator, entry),
        WorkClockButtonEditEntry(coordinator, entry),
        WorkClockButtonDeleteEntry(coordinator, entry),
        WorkClockButtonThisMonth(coordinator, entry),
        WorkClockButtonTodayEntry(coordinator, entry),
        WorkClockButtonClearStart(coordinator, entry),
        WorkClockButtonClearEnd(coordinator, entry),
    ]
    async_add_entities(entities)


class WorkClockButtonReload(WorkClockEntity, ButtonEntity):
    """Representation of work clock switch."""

    async def async_press(self) -> None:
        """Handle the button press."""
        self.coordinator.client.reload = True
        await self.coordinator.async_request_refresh()

    @property
    def unique_id(self):
        """Return a unique ID to use for this entity."""
        return self.config_entry.entry_id + "_reload"

    @property
    def name(self):
        """Return the name of the switch."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"{name.capitalize()} Reload"

    @property
    def icon(self):
        """Return the icon of the switch."""
        return "mdi:reload"


class WorkClockButtonDeleteState(WorkClockEntity, ButtonEntity):
    """Representation of work clock switch."""

    async def async_press(self) -> None:
        """Handle the button press."""
        await self.coordinator.client.async_delete_state()
        await self.coordinator.async_request_refresh()

    @property
    def unique_id(self):
        """Return a unique ID to use for this entity."""
        return self.config_entry.entry_id + "_delete_state"

    @property
    def name(self):
        """Return the name of the switch."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"{name.capitalize()} State Delete"

    @property
    def icon(self):
        """Return the icon of the switch."""
        return "mdi:timer-remove-outline"

    @property
    def extra_state_attributes(self) -> Mapping[str, Any] | None:
        """Return the state attributes."""
        my_attrs = {}
        for key in ("selected_state",):
            val = getattr(self.coordinator.client, key)
            if val is not None:
                val = val.strftime(DATE_FORMAT)
            my_attrs[key] = val
        attrs = super().extra_state_attributes
        if attrs is None:
            return my_attrs
        attrs.update(my_attrs)
        return attrs


class WorkClockButtonEditState(WorkClockEntity, ButtonEntity):
    """Representation of work clock switch."""

    async def async_press(self) -> None:
        """Handle the button press."""
        await self.coordinator.client.async_edit_state()
        await self.coordinator.async_request_refresh()

    @property
    def unique_id(self):
        """Return a unique ID to use for this entity."""
        return self.config_entry.entry_id + "_edit_state"

    @property
    def name(self):
        """Return the name of the switch."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"{name.capitalize()} State Edit"

    @property
    def icon(self):
        """Return the icon of the switch."""
        return "mdi:timer-edit-outline"

    @property
    def extra_state_attributes(self) -> Mapping[str, Any] | None:
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


class WorkClockButtonAddEntry(WorkClockEntity, ButtonEntity):
    """Representation of work clock switch."""

    async def async_press(self) -> None:
        """Handle the button press."""
        await self.coordinator.client.async_add_entry()
        await self.coordinator.async_request_refresh()

    @property
    def unique_id(self):
        """Return a unique ID to use for this entity."""
        return self.config_entry.entry_id + "_add_entry"

    @property
    def name(self):
        """Return the name of the switch."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"{name.capitalize()} Entry Add"

    @property
    def icon(self):
        """Return the icon of the switch."""
        return "mdi:calendar-plus"


class WorkClockButtonEditEntry(WorkClockEntity, ButtonEntity):
    """Representation of work clock switch."""

    async def async_press(self) -> None:
        """Handle the button press."""
        await self.coordinator.client.async_edit_entry()
        await self.coordinator.async_request_refresh()

    @property
    def unique_id(self):
        """Return a unique ID to use for this entity."""
        return self.config_entry.entry_id + "_edit_entry"

    @property
    def name(self):
        """Return the name of the switch."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"{name.capitalize()} Entry Edit"

    @property
    def icon(self):
        """Return the icon of the switch."""
        return "mdi:calendar-edit"


class WorkClockButtonDeleteEntry(WorkClockEntity, ButtonEntity):
    """Representation of work clock switch."""

    async def async_press(self) -> None:
        """Handle the button press."""
        await self.coordinator.client.async_delete_entry()
        await self.coordinator.async_request_refresh()

    @property
    def unique_id(self):
        """Return a unique ID to use for this entity."""
        return self.config_entry.entry_id + "_delete_entry"

    @property
    def name(self):
        """Return the name of the switch."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"{name.capitalize()} Entry Delete"

    @property
    def icon(self):
        """Return the icon of the switch."""
        return "mdi:calendar-remove"


class WorkClockButtonThisMonth(WorkClockEntity, ButtonEntity):
    """Representation of work clock switch."""

    async def async_press(self) -> None:
        """Handle the button press."""
        date_time = datetime.now(tz=self.coordinator.client.tz).replace(tzinfo=None)
        if date_time < self.coordinator.client.start_date:
            _LOGGER.warning(
                "Cannot select month before %s", date_time.strftime(MONTH_FORMAT)
            )
            return
        self.coordinator.client.selected_month = date_time
        self.coordinator.client.set_selected_entry(0)
        await self.coordinator.async_request_refresh()

    @property
    def unique_id(self):
        """Return a unique ID to use for this entity."""
        return self.config_entry.entry_id + "_this_month"

    @property
    def name(self):
        """Return the name of the switch."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"{name.capitalize()} Select This Month"

    @property
    def icon(self):
        """Return the icon of the switch."""
        return "mdi:table-arrow-left"


class WorkClockButtonTodayEntry(WorkClockEntity, ButtonEntity):
    """Representation of work clock switch."""

    async def async_press(self) -> None:
        """Handle the button press."""
        if self.coordinator.client.selected_month is None:
            return
        date_time = datetime.now(tz=self.coordinator.client.tz).replace(tzinfo=None)
        if (
            date_time.year != self.coordinator.client.selected_month.year
            or date_time.month != self.coordinator.client.selected_month.month
        ):
            return
        df_mon: pd.Series = self.coordinator.client.get_entries_month()["date"]
        mask = (
            (df_mon.dt.year == date_time.year)
            & (df_mon.dt.month == date_time.month)
            & (df_mon.dt.day == date_time.day)
        )
        if not any(mask):
            return
        self.coordinator.client.set_selected_entry(np.argmax(mask))
        await self.coordinator.async_request_refresh()

    @property
    def unique_id(self):
        """Return a unique ID to use for this entity."""
        return self.config_entry.entry_id + "_today_entry"

    @property
    def name(self):
        """Return the name of the switch."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"{name.capitalize()} Select Today Entry"

    @property
    def icon(self):
        """Return the icon of the switch."""
        return "mdi:calendar"


class WorkClockButtonClearStart(WorkClockEntity, ButtonEntity):
    """Representation of work clock switch."""

    async def async_press(self) -> None:
        """Handle the button press."""
        self.coordinator.client.new_t_start = None
        await self.coordinator.async_request_refresh()

    @property
    def unique_id(self):
        """Return a unique ID to use for this entity."""
        return self.config_entry.entry_id + "_clear_new_start"

    @property
    def name(self):
        """Return the name of the switch."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"{name.capitalize()} Clear New Start"


class WorkClockButtonClearEnd(WorkClockEntity, ButtonEntity):
    """Representation of work clock switch."""

    async def async_press(self) -> None:
        """Handle the button press."""
        self.coordinator.client.new_t_end = None
        await self.coordinator.async_request_refresh()

    @property
    def unique_id(self):
        """Return a unique ID to use for this entity."""
        return self.config_entry.entry_id + "_clear_new_end"

    @property
    def name(self):
        """Return the name of the switch."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"{name.capitalize()} Clear New End"
