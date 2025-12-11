from .registry import REGISTERED_PRESETS
from . import presets

try:
    import tensorpotential.experimental.presets
except ImportError:
    pass
from typing import Any


def get_preset(name: str):
    try:
        return REGISTERED_PRESETS[name]
    except KeyError:
        raise KeyError(
            f"Preset '{name}' if not found in the list of registered presets"
        )


def get_public_preset_list() -> list:
    default_list = []
    for name, preset in REGISTERED_PRESETS.items():
        if preset._public:
            default_list.append(name)
    return default_list


def get_default_preset_name() -> Any:
    names = get_public_preset_list()
    for name in names:
        if REGISTERED_PRESETS[name]._is_default:
            return name
    return None


def get_preset_settings(name: str) -> dict:
    return REGISTERED_PRESETS[name]._settings
