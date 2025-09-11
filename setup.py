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
        version="0.5.3",  ### UPD in pyproject.toml to avoid confusion
        scripts=[
            "bin/gracemaker",
            "bin/grace_models",
            "bin/grace_collect",
            "bin/extxyz2df",
            "bin/df2extxyz",
            "bin/grace_preprocess",
            "bin/grace_predict",
            "bin/grace_utils",
        ],
    )
