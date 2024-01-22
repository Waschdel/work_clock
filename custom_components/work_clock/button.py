"""Work Clock Buttons."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import logging
from typing import Any

import pandas as pd

from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import WorkClockDataUpdateCoordinator
from .const import DATE_FORMAT, DOMAIN
from .entity import WorkClockEntity

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Initialize WorkClock config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]
    entities = [
        WorkClockButtonDeleteState(coordinator, entry),
        WorkClockButtonEditState(coordinator, entry),
        WorkClockButtonThisMonth(coordinator, entry),
    ]
    async_add_entities(entities)


class WorkClockButtonDeleteState(WorkClockEntity, ButtonEntity):
    """Representation of work clock switch."""

    def __init__(
        self,
        coordinator: WorkClockDataUpdateCoordinator,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the WorkClock switch."""
        super().__init__(coordinator, config_entry)
        _LOGGER.info("Added %s", self.entity_id)

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
        return f"Delete {name.capitalize()} State"

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

    def __init__(
        self,
        coordinator: WorkClockDataUpdateCoordinator,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the WorkClock switch."""
        super().__init__(coordinator, config_entry)
        name: str = self.config_entry.options.get(CONF_NAME, "")
        self._attr_name = f"Edit {name.capitalize()} State"
        _LOGGER.info("Added %s", self.entity_id)

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
        return f"Edit {name.capitalize()} State"

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


class WorkClockButtonThisMonth(WorkClockEntity, ButtonEntity):
    """Representation of work clock switch."""

    def __init__(
        self,
        coordinator: WorkClockDataUpdateCoordinator,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the WorkClock switch."""
        super().__init__(coordinator, config_entry)
        _LOGGER.info("Added %s", self.entity_id)

    async def async_press(self) -> None:
        """Handle the button press."""
        date_time = pd.Timestamp(datetime.today()).round("d")
        if date_time < self.coordinator.client.start_date:
            _LOGGER.info(date_time)
            return
        self.coordinator.client.selected_month = date_time
        await self.coordinator.async_request_refresh()

    @property
    def unique_id(self):
        """Return a unique ID to use for this entity."""
        return self.config_entry.entry_id + "_this_month"

    @property
    def name(self):
        """Return the name of the switch."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"{name.capitalize()} This Month"

    @property
    def icon(self):
        """Return the icon of the switch."""
        return "mdi:table-arrow-left"
