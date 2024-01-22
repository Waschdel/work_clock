"""Work Clock Sensors."""
from __future__ import annotations

from datetime import timedelta
import logging

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import StateType

from . import WorkClockDataUpdateCoordinator
from .const import CONF_WORKHOURS, DOMAIN
from .entity import WorkClockEntity

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Initialize WorkClock config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]
    entities = [
        WorkClockSensorHours(coordinator, entry),
        WorkClockSensorEndTime(coordinator, entry),
    ]
    async_add_entities(entities)


class WorkClockSensorHours(WorkClockEntity, SensorEntity):
    """work_clock Sensor class."""

    def __init__(
        self,
        coordinator: WorkClockDataUpdateCoordinator,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the WorkClock switch."""
        super().__init__(coordinator, config_entry)
        self._attr_device_class = SensorDeviceClass.DURATION
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_native_unit_of_measurement = "h"
        _LOGGER.info("Added %s", self.entity_id)

    @property
    def unique_id(self):
        """Return a unique ID to use for this entity."""
        return self.config_entry.entry_id + "_hours"

    @property
    def name(self):
        """Return the name of the switch."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"{name.capitalize()} Hours"

    @property
    def icon(self):
        """Return the icon of the switch."""
        return "mdi:hours-24"

    @property
    def native_value(self) -> StateType:
        """Return the sensor value."""
        data = self.coordinator.client.states_day
        if not data.shape[0]:
            return 0
        return data.iloc[-1]["duration"].total_seconds() / 3600


class WorkClockSensorEndTime(WorkClockEntity, SensorEntity):
    """work_clock Sensor class."""

    def __init__(
        self,
        coordinator: WorkClockDataUpdateCoordinator,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the WorkClock switch."""
        super().__init__(coordinator, config_entry)
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
        self._attr_state_class = SensorStateClass.MEASUREMENT
        _LOGGER.info("Added %s", self.entity_id)

    @property
    def unique_id(self):
        """Return a unique ID to use for this entity."""
        return self.config_entry.entry_id + "_end_time"

    @property
    def name(self):
        """Return the name of the switch."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"{name.capitalize()} End Time"

    @property
    def icon(self):
        """Return the icon of the switch."""
        return "mdi:clock-end"

    @property
    def native_value(self) -> StateType:
        """Return the sensor value."""
        data = self.coordinator.client.states_day
        if not data.shape[0]:
            return None
        work_hours = self.config_entry.options.get(CONF_WORKHOURS, 7)
        return data.iloc[-1]["start"] + timedelta(hours=work_hours)
