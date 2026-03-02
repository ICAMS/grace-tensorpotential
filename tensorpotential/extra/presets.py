import importlib
from pathlib import Path


def load_extra_models():
    current_dir = Path(__file__).parent
    for file_path in current_dir.rglob("*.py"):
        if "model" in file_path.name:
            relative_path = file_path.relative_to(current_dir)
            module_parts = relative_path.with_suffix("").parts
            module_name = "tensorpotential.extra." + ".".join(module_parts)
            try:
                importlib.import_module(module_name)
            except ImportError:
                pass


load_extra_models()
