"""Work Clock Selects."""
from __future__ import annotations

from datetime import datetime
import logging

import voluptuous as vol

from homeassistant.components.select import SelectEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import (
    AddEntitiesCallback,
    async_get_current_platform,
)

from . import WorkClockDataUpdateCoordinator
from .const import ATTR_DATETIME, DATE_FORMAT, DOMAIN, MONTH_FORMAT
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
    ]
    async_add_entities(entities)

    # add services
    platform = async_get_current_platform()
    platform.async_register_entity_service(
        "delete_state",
        {},
        "async_delete_state",
    )
    platform.async_register_entity_service(
        "edit_state",
        vol.Schema({vol.Required(ATTR_DATETIME): cv.datetime}, extra=vol.ALLOW_EXTRA),
        "async_edit_state",
    )


class WorkClockSelectState(WorkClockEntity, SelectEntity):
    """Representation of work clock state select."""

    def __init__(
        self,
        coordinator: WorkClockDataUpdateCoordinator,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the WorkClock switch."""
        super().__init__(coordinator, config_entry)
        self._attr_current_option = None
        _LOGGER.info("Added %s", self.entity_id)

    async def async_select_option(self, option: str) -> None:
        """Select new item (option)."""
        s_date_time = option.split(", ")[0]
        date_time = datetime.strptime(s_date_time, DATE_FORMAT)
        self.coordinator.client.selected_state = date_time
        self.coordinator.client.new_state = date_time
        self._attr_current_option = option
        await self.coordinator.async_request_refresh()

    async def async_delete_state(self) -> bool:
        """Delete state from states."""
        if self._attr_current_option is None:
            _LOGGER.error("No state selected")
            return False
        if self._attr_current_option not in self.options:
            _LOGGER.error("'%s' not found", self._attr_current_option)
            return False

        res = await self.coordinator.client.async_delete_state()
        await self.coordinator.async_request_refresh()
        return res

    async def async_edit_state(self) -> bool:
        """Edit state datetime in states."""
        return False

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
    def options(self) -> list[str]:
        """Return the available items."""
        options = self.coordinator.client.states.iloc[:30]
        if not options.shape[0]:
            return []
        return options.apply(
            lambda x: "%s, %d" % (x["datetime"].strftime(DATE_FORMAT), x["state"]),
            axis=1,
        ).to_list()


class WorkClockSelectMonth(WorkClockEntity, SelectEntity):
    """Representation of work clock state select."""

    def __init__(
        self,
        coordinator: WorkClockDataUpdateCoordinator,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the WorkClock switch."""
        super().__init__(coordinator, config_entry)
        self._attr_current_option = None
        _LOGGER.info("Added %s", self.entity_id)

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
        return date_time.strftime(MONTH_FORMAT)

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
        self._attr_current_option = None
        _LOGGER.info("Added %s", self.entity_id)

    async def async_select_option(self, option: str) -> None:
        """Select new item (option)."""
        try:
            self.coordinator.client.selected_entry = int(option.split(" ")[-1])
        except ValueError:
            _LOGGER.error("COuld not get index from %s", option)
            return
        _LOGGER.error("%d is %s", self.coordinator.client.selected_entry, option)
        self._attr_current_option = option
        await self.coordinator.async_request_refresh()

    async def async_delete_state(self) -> bool:
        """Delete state from states."""
        if self._attr_current_option is None:
            _LOGGER.error("No state selected")
            return False
        if self._attr_current_option not in self.options:
            _LOGGER.error("'%s' not found", self._attr_current_option)
            return False

        res = await self.coordinator.client.async_delete_state()
        await self.coordinator.async_request_refresh()
        return res

    async def async_edit_state(self) -> bool:
        """Edit state datetime in states."""
        return False

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
    def options(self) -> list[str]:
        """Return the available items."""
        if self.coordinator.client.selected_month is None:
            return []
        y = self.coordinator.client.selected_month.year
        m = self.coordinator.client.selected_month.month
        mask = (self.coordinator.client.entries["date"].dt.year == y) & (
            self.coordinator.client.entries["date"].dt.month == m
        )
        if not any(mask):
            return []
        options = self.coordinator.client.entries.loc[mask, ["date", "type"]]
        options["i"] = list(range(options.shape[0]))

        return options.apply(
            lambda x: f'{x["date"].strftime("%d.%m.%Y %a")} {x["type"]} {x["i"]}',
            axis=1,
        ).to_list()
