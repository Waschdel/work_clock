"""Work Clock Selects."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import fnmatch
import logging
from typing import Any

import pandas as pd

from homeassistant.components.select import SelectEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import WorkClockDataUpdateCoordinator
from .const import DATE_FORMAT, DOMAIN, ENTRY_TYPES, ENTRY_TYPES_OFF, MONTH_FORMAT
from .entity import WorkClockEntity

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Initialize WorkClock config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]
    entities = [
        WorkClockSelectState(coordinator, entry),
        WorkClockSelectMonth(coordinator, entry),
        WorkClockSelectEntry(coordinator, entry),
        WorkClockSelectNewType(coordinator, entry),
    ]
    async_add_entities(entities)


class WorkClockSelectState(WorkClockEntity, SelectEntity):
    """Representation of work clock state select."""

    async def async_select_option(self, option: str) -> None:
        """Select new item (option)."""
        s_date_time = option.split(", ")[0]
        date_time = datetime.strptime(s_date_time, DATE_FORMAT)
        self.coordinator.client.selected_state = date_time
        self.coordinator.client.new_state = date_time
        await self.coordinator.async_request_refresh()

    @property
    def unique_id(self):
        """Return a unique ID to use for this entity."""
        return self.config_entry.entry_id + "_state"

    @property
    def name(self):
        """Return the name of the switch."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"Select {name.capitalize()} State"

    @property
    def icon(self):
        """Return the icon of the switch."""
        return "mdi:timer-check-outline"

    @property
    def current_option(self) -> str | None:
        """Return the selected entity option to represent the entity state."""
        if self.coordinator.client.selected_state is None:
            return None
        date_str = self.coordinator.client.selected_state.strftime(DATE_FORMAT)
        options = fnmatch.filter(self.options, date_str + "*")
        if len(options) != 1:
            return None
        if options[0] not in self.options:
            return None
        return options[0]

    @property
    def options(self) -> list[str]:
        """Return the available items."""
        options = self.coordinator.client.states.iloc[:30]
        if not options.shape[0]:
            return []
        return options.apply(
            lambda x: "%s, %d"
            % (
                x["datetime"]
                .astimezone(self.coordinator.client.tz)
                .strftime(DATE_FORMAT),
                x["state"],
            ),
            axis=1,
        ).to_list()


class WorkClockSelectMonth(WorkClockEntity, SelectEntity):
    """Representation of work clock state select."""

    async def async_select_option(self, option: str) -> None:
        """Select new item (option)."""
        try:
            date_time = datetime.strptime(option, MONTH_FORMAT)
        except ValueError:
            _LOGGER.error("Unknown option %s", option)
            return
        self.coordinator.client.selected_month = date_time
        await self.coordinator.async_request_refresh()

    @property
    def unique_id(self):
        """Return a unique ID to use for this entity."""
        return self.config_entry.entry_id + "_month"

    @property
    def name(self):
        """Return the name of the switch."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"Select {name.capitalize()} Month"

    @property
    def icon(self):
        """Return the icon of the switch."""
        return "mdi:timer-check-outline"

    @property
    def current_option(self) -> str | None:
        """Return the selected entity option to represent the entity state."""
        date_time = self.coordinator.client.selected_month
        if date_time is None:
            return None
        option = date_time.strftime(MONTH_FORMAT)
        if option not in self.options:
            return
        return option

    @property
    def options(self) -> list[str]:
        """Return the available items."""
        options = (
            self.coordinator.client.entries["date"]
            .dt.strftime(MONTH_FORMAT)
            .drop_duplicates()
        )
        if not options.shape[0]:
            return []
        return options.to_list()


class WorkClockSelectEntry(WorkClockEntity, SelectEntity):
    """Representation of work clock state select."""

    def __init__(
        self,
        coordinator: WorkClockDataUpdateCoordinator,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the WorkClock switch."""
        super().__init__(coordinator, config_entry)
        _LOGGER.debug("Added %s", self.entity_id)

    async def async_select_option(self, option: str) -> None:
        """Select new item (option)."""
        if option not in self.options:
            return
        try:
            i_entry = int(option.split(" ")[-1])
        except ValueError:
            _LOGGER.error("COuld not get index from %s", option)
            return
        _LOGGER.debug("%d is %s", i_entry, option)
        self.coordinator.client.selected_entry = i_entry
        row = self.coordinator.client.get_selected_entry().iloc[0]
        if row is not None:
            self.coordinator.client.new_type = (
                None if row.isna()["type"] else row["type"]
            )
            d = row["date"]
            if row.isna()["start"]:
                self.coordinator.client.new_t_start = None
            else:
                start: pd.Timestamp = row["start"]
                start.replace(year=d.year, month=d.month, day=d.day)
                self.coordinator.client.new_t_start = start.to_pydatetime()
            if row.isna()["end"]:
                self.coordinator.client.new_t_end = None
            else:
                t_end: pd.Timestamp = row["end"]
                t_end.replace(year=d.year, month=d.month, day=d.day)
                self.coordinator.client.new_t_end = t_end.to_pydatetime()
        await self.coordinator.async_request_refresh()

    @property
    def unique_id(self):
        """Return a unique ID to use for this entity."""
        return self.config_entry.entry_id + "_entry"

    @property
    def name(self):
        """Return the name of the switch."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"Select {name.capitalize()} Entry"

    @property
    def icon(self):
        """Return the icon of the switch."""
        return "mdi:timer-check-outline"

    @property
    def current_option(self) -> str | None:
        """Return the selected entity option to represent the entity state."""
        i = self.coordinator.client.selected_entry
        if i is None:
            return
        options = self.options
        if options is None:
            return
        options = fnmatch.filter(options, f"* {i}")
        if not options:
            return
        if options[0] not in self.options:
            return
        return options[0]

    @property
    def options(self) -> list[str]:
        """Return the available items."""
        rows = self.coordinator.client.get_entries_month()
        if rows is None or not rows.shape[0]:
            return
        options = rows.loc[:, ["date", "type"]]
        options["i"] = list(range(options.shape[0]))
        return options.apply(
            lambda x: f'{x["date"].strftime("%d.%m.%Y %a")} {x["type"]} {x["i"]}',
            axis=1,
        ).to_list()


class WorkClockSelectNewType(WorkClockEntity, SelectEntity):
    """Representation of work clock state select."""

    async def async_select_option(self, option: str) -> None:
        """Select new item (option)."""
        if option not in self.options:
            return
        self.coordinator.client.new_type = option
        if option in ENTRY_TYPES_OFF:
            self.coordinator.client.new_t_start = None
            self.coordinator.client.new_t_end = None
        await self.coordinator.async_request_refresh()

    @property
    def unique_id(self):
        """Return a unique ID to use for this entity."""
        return self.config_entry.entry_id + "_new_type"

    @property
    def name(self):
        """Return the name of the switch."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"Select New {name.capitalize()} Type"

    @property
    def icon(self):
        """Return the icon of the switch."""
        return "mdi:timer-check-outline"

    @property
    def current_option(self) -> str | None:
        """Return the selected entity option to represent the entity state."""
        t = self.coordinator.client.new_type
        if t is None or t not in self.options:
            return
        return t

    @property
    def options(self) -> list[str]:
        """Return the available items."""
        return ENTRY_TYPES

    @property
    def extra_state_attributes(self) -> Mapping[str, Any] | None:
        """Return the state attributes."""
        attrs = super().extra_state_attributes
        row = self.coordinator.client.get_selected_entry()
        if row is None:
            return attrs
        my_attrs = {"old": row["type"].iloc[0]}
        if attrs is None:
            return my_attrs
        attrs.update(my_attrs)
        return attrs
