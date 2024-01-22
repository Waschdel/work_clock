"""Work Clock Switches."""
from __future__ import annotations

import logging
from typing import Literal

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
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
    async_add_entities([WorkClockSwitch(coordinator, entry)])


class WorkClockSwitch(WorkClockEntity, SwitchEntity):
    """Representation of work clock switch."""

    def __init__(
        self,
        coordinator: WorkClockDataUpdateCoordinator,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the WorkClock switch."""
        super().__init__(coordinator, config_entry)
        self.entity_id = self.coordinator.client.entity_id

    async def async_turn_on(self, **kwargs) -> None:
        """Turn the device on."""
        await self.coordinator.client.async_write_state(True)
        await self.coordinator.async_request_refresh()

    async def async_turn_off(self, **kwargs) -> None:
        """Turn the device off."""
        await self.coordinator.client.async_write_state(False)
        await self.coordinator.async_request_refresh()

    @property
    def name(self):
        """Return the name of the switch."""
        name: str = self.config_entry.options.get(CONF_NAME, "")
        return f"{name.capitalize()} State"

    @property
    def is_on(self) -> bool | None:
        """Return True if entity is on."""
        return self.coordinator.client.is_on

    @property
    def assumed_state(self) -> Literal[False]:
        """Assumed state is off."""
        return False
