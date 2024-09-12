from setuptools import setup, find_packages

setup(
    name="tensorpotential",
    version="0.4.0",
    packages=find_packages(include=["tensorpotential", "tensorpotential.*"]),
    url="",
    license="",
    author="Anton Bochkarev, Yury Lysogorskiy",
    author_email="",
    description="",
    python_requires="<3.12",
    install_requires=[
        "scipy",
        "tensorflow<2.17",
        "matscipy",
        "numpy",
        "sympy",
        "pandas<3.0.0",
        "ase",
        "pyyaml",
        "tqdm",
    ],
    scripts=[
        "bin/gracemaker",
        "bin/grace_download",
    ],
)
