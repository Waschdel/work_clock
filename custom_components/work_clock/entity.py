"""WorkClockEntity class."""
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from . import WorkClockDataUpdateCoordinator
from .const import DOMAIN, ICON, NAME, VERSION


class WorkClockEntity(CoordinatorEntity):
    """Work clock template entity."""

    def __init__(
        self, coordinator: WorkClockDataUpdateCoordinator, config_entry: ConfigEntry
    ) -> None:
        """Init Work clock template entity."""
        super().__init__(coordinator)
        self.config_entry = config_entry

    @property
    def unique_id(self):
        """Return a unique ID to use for this entity."""
        return self.config_entry.entry_id

    @property
    def device_info(self):
        """Return the device info."""
        return {
            "identifiers": {(DOMAIN, self.unique_id)},
            "name": NAME,
            "model": VERSION,
            "manufacturer": NAME,
        }

    @property
    def device_state_attributes(self):
        """Return the state attributes."""
        return {
            "id": str(self.coordinator.data.get("id")),
            "integration": DOMAIN,
        }

    @property
    def icon(self):
        """Return icon to be used for this sensor."""
        return ICON
