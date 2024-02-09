"""Work Clock Sensors."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
import logging
from typing import Any

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
from .const import CONF_WORKHOURS, DOMAIN, ENTRY_TYPES_K
from .entity import WorkClockEntity
from .util import first_of_month, last_of_month

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Initialize WorkClock config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]
    entities = [
        WorkClockSensorHours(coordinator, entry),
        WorkClockSensorEndTime(coordinator, entry),
        WorkClockSensorMonFG(coordinator, entry),
        WorkClockSensorMonFZ(coordinator, entry),
        WorkClockSensorMonU(coordinator, entry),
        WorkClockSensorMonK(coordinator, entry),
        WorkClockSensorMonMA(coordinator, entry),
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
        self._attr_native_unit_of_measurement = "h"

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
        return round(data.iloc[-1]["duration"].total_seconds() / 3600, 2)


class WorkClockSensorEndTime(WorkClockEntity, SensorEntity):
    """work_clock Sensor class."""

    def __init__(
        self,
        coordinator: WorkClockDataUpdateCoordinator,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the WorkClock switch."""
        super().__init__(coordinator, config_entry)
        self._attr_device_class = None
        self._attr_state_class = None

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
        t_end = data.iloc[-1]["start"] + timedelta(hours=work_hours)
        return t_end.strftime("%H:%M")


class WorkClockSensorMonFG(WorkClockEntity, SensorEntity):
    """work_clock Sensor class."""

    def __init__(
        self,
        coordinator: WorkClockDataUpdateCoordinator,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the WorkClock switch."""
        super().__init__(coordinator, config_entry)
        self._attr_device_class = None
        self._attr_native_unit_of_measurement = "h"

    @property
    def unique_id(self):
        """Return a unique ID to use for this entity."""
        return self.config_entry.entry_id + "_fg"

    @property
    def name(self):
        """Return the name of the switch."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"{name.capitalize()} FG"

    @property
    def icon(self):
        """Return the icon of the switch."""
        return "mdi:hours-24"

    @property
    def native_value(self) -> StateType:
        """Return the sensor value."""
        if self.coordinator.client.selected_month is None:
            return
        attr = self.extra_state_attributes
        if "end" not in attr:
            return
        return attr["end"]

    @property
    def extra_state_attributes(self) -> Mapping[str, Any] | None:
        """Return the state attributes."""
        attrs = super().extra_state_attributes
        if self.coordinator.client.selected_month is None:
            return attrs
        d0 = first_of_month(self.coordinator.client.selected_month)
        d1 = last_of_month(self.coordinator.client.selected_month) + timedelta(days=1)
        fgs = [
            self.coordinator.client.entries.loc[
                self.coordinator.client.entries["date"] < d, "overhours"
            ].sum()
            + self.coordinator.client.start_fg
            for d in [d0, d1]
        ]
        my_attrs = {
            "start": round(fgs[0], 2),
            "end": round(fgs[1], 2),
            "delta": round(fgs[1] - fgs[0], 2),
        }
        if attrs is None:
            return my_attrs
        attrs.update(my_attrs)
        return attrs


class WorkClockSensorMonFZ(WorkClockEntity, SensorEntity):
    """work_clock Sensor class."""

    def __init__(
        self,
        coordinator: WorkClockDataUpdateCoordinator,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the WorkClock switch."""
        super().__init__(coordinator, config_entry)
        self._attr_device_class = None
        self._attr_native_unit_of_measurement = "h"

    @property
    def unique_id(self):
        """Return a unique ID to use for this entity."""
        return self.config_entry.entry_id + "_fz"

    @property
    def name(self):
        """Return the name of the switch."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"{name.capitalize()} FZ"

    @property
    def icon(self):
        """Return the icon of the switch."""
        return "mdi:hours-24"

    @property
    def native_value(self) -> StateType:
        """Return the sensor value."""
        if self.coordinator.client.selected_month is None:
            return
        attr = self.extra_state_attributes
        if "end" not in attr:
            return
        return attr["end"]

    @property
    def extra_state_attributes(self) -> Mapping[str, Any] | None:
        """Return the state attributes."""
        attrs = super().extra_state_attributes
        if self.coordinator.client.selected_month is None:
            return attrs
        d0 = first_of_month(self.coordinator.client.selected_month) - timedelta(days=1)
        d1 = last_of_month(self.coordinator.client.selected_month)
        fgs = [
            self.coordinator.client.calc_fz(d).total_seconds() / 3600 for d in [d0, d1]
        ]
        my_attrs = {
            "start": round(fgs[0], 2),
            "end": round(fgs[1], 2),
            "delta": round(fgs[1] - fgs[0], 2),
        }
        if attrs is None:
            return my_attrs
        attrs.update(my_attrs)
        return attrs


class WorkClockSensorMonU(WorkClockEntity, SensorEntity):
    """work_clock Sensor class."""

    def __init__(
        self,
        coordinator: WorkClockDataUpdateCoordinator,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the WorkClock switch."""
        super().__init__(coordinator, config_entry)
        self._attr_device_class = None
        self._attr_native_unit_of_measurement = "d"

    @property
    def unique_id(self):
        """Return a unique ID to use for this entity."""
        return self.config_entry.entry_id + "_u"

    @property
    def name(self):
        """Return the name of the switch."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"{name.capitalize()} U"

    @property
    def icon(self):
        """Return the icon of the switch."""
        return "mdi:hours-24"

    @property
    def native_value(self) -> StateType:
        """Return the sensor value."""
        if self.coordinator.client.selected_month is None:
            return
        attr = self.extra_state_attributes
        if "year" not in attr:
            return
        return attr["year"]

    @property
    def extra_state_attributes(self) -> Mapping[str, Any] | None:
        """Return the state attributes."""
        attrs = super().extra_state_attributes
        if self.coordinator.client.selected_month is None:
            return attrs
        y0 = datetime(self.coordinator.client.selected_month.year, 1, 1)
        d0 = first_of_month(self.coordinator.client.selected_month)
        d1 = last_of_month(self.coordinator.client.selected_month)
        us = [
            self.coordinator.client.entries.loc[
                (self.coordinator.client.entries["date"] >= d)
                & (self.coordinator.client.entries["date"] <= d1)
                & (self.coordinator.client.entries["type"] == "U"),
            ].shape[0]
            for d in [y0, d0]
        ]
        my_attrs = {
            "year": us[0],
            "month": us[1],
        }
        if attrs is None:
            return my_attrs
        attrs.update(my_attrs)
        return attrs


class WorkClockSensorMonK(WorkClockEntity, SensorEntity):
    """work_clock Sensor class."""

    def __init__(
        self,
        coordinator: WorkClockDataUpdateCoordinator,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the WorkClock switch."""
        super().__init__(coordinator, config_entry)
        self._attr_device_class = None
        self._attr_native_unit_of_measurement = "d"

    @property
    def unique_id(self):
        """Return a unique ID to use for this entity."""
        return self.config_entry.entry_id + "_k"

    @property
    def name(self):
        """Return the name of the switch."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"{name.capitalize()} K"

    @property
    def icon(self):
        """Return the icon of the switch."""
        return "mdi:hours-24"

    @property
    def native_value(self) -> StateType:
        """Return the sensor value."""
        if self.coordinator.client.selected_month is None:
            return
        attr = self.extra_state_attributes
        return sum([attr.get(k, 0) for k in ENTRY_TYPES_K])

    @property
    def extra_state_attributes(self) -> Mapping[str, Any] | None:
        """Return the state attributes."""
        attrs = super().extra_state_attributes
        if self.coordinator.client.selected_month is None:
            return attrs
        y0 = datetime(self.coordinator.client.selected_month.year, 1, 1)
        d1 = last_of_month(self.coordinator.client.selected_month)
        my_attrs = {
            k: self.coordinator.client.entries.loc[
                (self.coordinator.client.entries["date"] >= y0)
                & (self.coordinator.client.entries["date"] <= d1)
                & (self.coordinator.client.entries["type"] == k),
            ].shape[0]
            for k in ENTRY_TYPES_K
        }
        if attrs is None:
            return my_attrs
        attrs.update(my_attrs)
        return attrs


class WorkClockSensorMonMA(WorkClockEntity, SensorEntity):
    """work_clock Sensor class."""

    def __init__(
        self,
        coordinator: WorkClockDataUpdateCoordinator,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the WorkClock switch."""
        super().__init__(coordinator, config_entry)
        self._attr_device_class = None
        self._attr_native_unit_of_measurement = "h"

    @property
    def unique_id(self):
        """Return a unique ID to use for this entity."""
        return self.config_entry.entry_id + "_ma"

    @property
    def name(self):
        """Return the name of the switch."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"{name.capitalize()} MA"

    @property
    def icon(self):
        """Return the icon of the switch."""
        return "mdi:hours-24"

    @property
    def native_value(self) -> StateType:
        """Return the sensor value."""
        if self.coordinator.client.selected_month is None:
            return
        d0 = first_of_month(self.coordinator.client.selected_month)
        d1 = last_of_month(self.coordinator.client.selected_month)
        ma = self.coordinator.client.entries.loc[
            (self.coordinator.client.entries["date"] >= d0)
            & (self.coordinator.client.entries["date"] <= d1)
            & (self.coordinator.client.entries["type"] == "MA"),
            "time_booked",
        ].sum()
        return round(ma, 2)
