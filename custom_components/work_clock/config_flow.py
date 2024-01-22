"""Config flow for Work Clock integration."""
from __future__ import annotations

import logging
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy.exc import SQLAlchemyError
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.components.recorder import CONF_DB_URL, get_instance, validate_db_url
from homeassistant.components.sql.util import resolve_db_url
from homeassistant.const import CONF_NAME, CONF_TIME_ZONE
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector
from homeassistant.helpers.schema_config_entry_flow import SchemaFlowError

from .const import CONF_START_DATE, CONF_START_FG, CONF_START_FZ, CONF_WORKHOURS, DOMAIN

_LOGGER = logging.getLogger(__name__)


def _validate_config(data: Any) -> Any:
    """Validate config."""
    try:
        validate_db_url(data[CONF_DB_URL])
    except vol.Invalid as exc:
        raise SchemaFlowError("URL not valid") from exc

    return data


OPTIONS_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_DB_URL): selector.TextSelector(),
        vol.Required(CONF_TIME_ZONE): selector.TextSelector(),
        vol.Required(CONF_WORKHOURS, default=7): selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=0,
                max=24,
                mode=selector.NumberSelectorMode.BOX,
                unit_of_measurement="hours",
            ),
        ),
        vol.Required(CONF_START_DATE): selector.DateSelector(),
        vol.Required(CONF_START_FG): selector.NumberSelector(
            selector.NumberSelectorConfig(
                mode=selector.NumberSelectorMode.BOX,
                step=0.01,
                unit_of_measurement="hours",
            )
        ),
        vol.Required(CONF_START_FZ): selector.NumberSelector(
            selector.NumberSelectorConfig(
                mode=selector.NumberSelectorMode.BOX,
                step=0.01,
                unit_of_measurement="hours",
            )
        ),
    }
)

CONFIG_SCHEMA: vol.Schema = vol.Schema(
    {
        vol.Required(CONF_NAME, default="work"): selector.TextSelector(),
    }
).extend(OPTIONS_SCHEMA.schema)


class WorkClockConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Work Clock config flow."""

    VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> WorkClockOptionsFlowHandler:
        """Get the options flow for this handler."""
        return WorkClockOptionsFlowHandler(config_entry)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the user step."""
        errors = {}
        description_placeholders = {}

        if user_input is not None:
            db_url = user_input.get(CONF_DB_URL)
            db_url_for_validation = None

            try:
                db_url_for_validation = resolve_db_url(self.hass, db_url)
            except SQLAlchemyError:
                errors["db_url"] = "db_url_invalid"

            try:
                ZoneInfo(user_input[CONF_TIME_ZONE])
            except FileNotFoundError:
                errors["timezone"] = "timezone_invalid"

            options = {
                CONF_NAME: user_input[CONF_NAME],
                CONF_TIME_ZONE: user_input[CONF_TIME_ZONE],
                CONF_WORKHOURS: user_input[CONF_WORKHOURS],
                CONF_START_DATE: user_input[CONF_START_DATE],
                CONF_START_FG: user_input[CONF_START_FG],
                CONF_START_FZ: user_input[CONF_START_FZ],
            }
            if db_url_for_validation != get_instance(self.hass).db_url:
                options[CONF_DB_URL] = db_url_for_validation

            if not errors:
                return self.async_create_entry(
                    title=user_input[CONF_NAME],
                    data={},
                    options=options,
                )

        return self.async_show_form(
            step_id="user",
            data_schema=self.add_suggested_values_to_schema(CONFIG_SCHEMA, user_input),
            errors=errors,
            description_placeholders=description_placeholders,
        )


class WorkClockOptionsFlowHandler(config_entries.OptionsFlowWithConfigEntry):
    """WorkClock options."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage WorkClock options."""
        errors = {}
        description_placeholders = {}

        if user_input is not None:
            db_url = user_input.get(CONF_DB_URL)
            name = self.options.get(CONF_NAME, self.config_entry.title)

            try:
                db_url_for_validation = resolve_db_url(self.hass, db_url)
            except SQLAlchemyError:
                errors["db_url"] = "db_url_invalid"
            else:
                recorder_db = get_instance(self.hass).db_url
                _LOGGER.debug(
                    "db_url: %s, resolved db_url: %s, recorder: %s",
                    db_url,
                    db_url_for_validation,
                    recorder_db,
                )

                options = {
                    CONF_NAME: name,
                    CONF_TIME_ZONE: user_input[CONF_TIME_ZONE],
                    CONF_WORKHOURS: user_input[CONF_WORKHOURS],
                    CONF_START_DATE: user_input[CONF_START_DATE],
                    CONF_START_FG: user_input[CONF_START_FG],
                    CONF_START_FZ: user_input[CONF_START_FZ],
                }
                if db_url_for_validation != get_instance(self.hass).db_url:
                    options[CONF_DB_URL] = db_url_for_validation

                return self.async_create_entry(
                    data=options,
                )

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                OPTIONS_SCHEMA, user_input or self.options
            ),
            errors=errors,
            description_placeholders=description_placeholders,
        )
