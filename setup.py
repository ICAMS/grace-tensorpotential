from setuptools import setup, find_packages

# (Optional) Version reading from pyproject.toml
# import pathlib
# import toml
#
# HERE = pathlib.Path(__file__).parent
# TOML_PATH = HERE / "pyproject.toml"
# TOML_DATA = toml.load(TOML_PATH)
# VERSION = TOML_DATA["project"]["version"]


if __name__ == "__main__":
    setup(
        # version=VERSION # (Optional) when version is read from pyproject.toml
        packages=find_packages(include=["tensorpotential", "tensorpotential.*"]),
        package_data={"tensorpotential": ["resources/input_template.yaml"]},
        include_package_data=True,
        # version is now read from pyproject.toml
    )
