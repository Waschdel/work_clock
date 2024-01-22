"""Work Clock Binary Sensors."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
import logging
from typing import Any

import numpy as np
import pandas as pd

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import StateType

from . import WorkClockDataUpdateCoordinator
from .const import DOMAIN
from .entity import WorkClockEntity
from .util import format_timedelta

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Initialize WorkClock config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]
    entities = [WorkClockSensorEntry(coordinator, entry, i) for i in range(50)]
    async_add_entities(entities)


class WorkClockSensorEntry(WorkClockEntity, BinarySensorEntity):
    """work_clock Sensor class."""

    def __init__(
        self,
        coordinator: WorkClockDataUpdateCoordinator,
        config_entry: ConfigEntry,
        nr: int,
    ) -> None:
        """Initialize the WorkClock switch."""
        super().__init__(coordinator, config_entry)
        self._attr_device_class = BinarySensorDeviceClass.PRESENCE
        self._attr_state_class = None
        self.nr: int = nr

    @property
    def unique_id(self):
        """Return a unique ID to use for this entity."""
        return f"{self.config_entry.entry_id}_entry{self.nr}"

    @property
    def name(self):
        """Return the name of the switch."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"{name.capitalize()} Entry {self.nr}"

    @property
    def icon(self):
        """Return the icon of the switch."""
        return "mdi:calendar-today"

    @property
    def is_on(self) -> bool | None:
        """Return true if the binary sensor is on."""
        if self.coordinator.client.selected_entry is None:
            return False
        return self.nr == self.coordinator.client.selected_entry

    @property
    def extra_state_attributes(self) -> Mapping[str, Any] | None:
        """Return the state attributes."""
        attrs = super().extra_state_attributes
        rows = self.coordinator.client.get_entries_month()
        if rows is None or rows.shape[0] <= self.nr:
            return attrs
        row = rows.iloc[self.nr]
        mask_na = rows.isna().iloc[self.nr]
        calc = (
            ~mask_na["type"]
            & row["calculated"]
            & (row["date"] < datetime.now() - timedelta(days=1))
        )
        my_attrs = {
            "date": row["date"].strftime("%d.%m.%Y"),
            "day": row["date"].strftime("%a"),
            "calculated": str(calc),
        }
        for key in rows.columns:
            if key in my_attrs:
                continue
            if mask_na[key]:
                my_attrs[key] = ""
            elif isinstance(row[key], datetime):
                my_attrs[key] = row[key].strftime("%H:%M")
            elif isinstance(row[key], timedelta):
                my_attrs[key] = format_timedelta(row[key])
            elif isinstance(row[key], float):
                my_attrs[key] = "%.2f" % row[key]
            else:
                my_attrs[key] = str(row[key])
        attrs = super().extra_state_attributes
        if attrs is None:
            return my_attrs
        attrs.update(my_attrs)
        return attrs
