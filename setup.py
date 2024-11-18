from setuptools import setup, find_packages

setup(
    name="tensorpotential",
    version="0.4.3",
    packages=find_packages(include=["tensorpotential", "tensorpotential.*"]),
    url="https://github.com/ICAMS/grace-tensorpotential",
    license="Academic Software License (ASL)",
    author="Anton Bochkarev, Yury Lysogorskiy",
    author_email="anton.bochkarev@rub.de, yury.lysogorskiy@rub.de",
    description="Graph Atomic Cluster Expansion (GRACE)",
    long_description="Graph Atomic Cluster Expansion (GRACE)",
    python_requires="<3.12",
    install_requires=[
        "scipy",
        "tensorflow<2.17",
        "matscipy",
        "numpy",
        "sympy",
        "pandas<3.0.0",
        "ase",
        "pyyaml>=6.0.2",
        "tqdm",
    ],
    scripts=[
        "bin/gracemaker",
        "bin/grace_models",
        "bin/grace_collect",
        "bin/extxyz2df",
    ],
    package_data={"tensorpotential": ["resources/input_template.yaml"]},
    include_package_data=True,
)
