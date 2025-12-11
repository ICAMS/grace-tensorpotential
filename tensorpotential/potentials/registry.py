REGISTERED_PRESETS: dict = {}


def register_preset(
    name: str, settings: dict = None, default: bool = False, public: bool = False
):
    def decorator(obj):
        obj._settings = settings
        obj._is_default = default
        obj._public = public
        REGISTERED_PRESETS[name] = obj
        return obj

    if default:
        assert not any(
            f._is_default for f in REGISTERED_PRESETS.values()
        ), "Can not have multiple default options for preset name"

    return decorator
