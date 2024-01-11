from setuptools import setup, find_packages

setup(
    name="RIGID.py",
    version="0.0.1",
    description="Geometry optimization of an atomic system split into rigid fragments",
    packages=find_packages(),
    license="MIT",
    python_requires=">3.8.0",
    url="https://github.com/siegfriedkaidisch/RIGID.py",
    author="Siegfried Kaidisch",
    author_email="siegfried.kaidisch@uni-graz.at",
    install_requires=["numpy", "ase"],
    extras_require={"dev": ["twine", "wheel"]},
    include_package_data=True,
)
