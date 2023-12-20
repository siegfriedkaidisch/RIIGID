from setuptools import setup, find_packages

setup(
    name="RIGID",
    version="0.0.1a",
    description="Geometry optimization of rigid molecules on rigid surfaces.",
    packages=find_packages(),
    license="MIT",
    python_requires=">3.8.0",
    url="https://github.com/siegfriedkaidisch/RIGID",
    author="Siegfried Kaidisch",
    author_email="siegfried.kaidisch@uni-graz.at",
    install_requires=["numpy", "ase"],
    extras_require={"dev": ["twine", "wheel"]},
    include_package_data=True,
)
