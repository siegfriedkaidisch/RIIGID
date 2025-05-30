from setuptools import setup, find_packages

setup(
    name="RIIGID",
    version="1.2.0",
    description="RIIGID - RIgid Interface Geometry IDentification",
    packages=find_packages(),
    license="MIT",
    python_requires=">3.9.0",
    url="https://github.com/siegfriedkaidisch/RIIGID",
    author="Siegfried Kaidisch",
    author_email="siegfried.kaidisch@uni-graz.at",
    install_requires=["numpy", "ase", "scikit-learn"],
    extras_require={"dev": ["twine", "wheel"]},
    include_package_data=True,
    package_data={'riigid': ['config.json']},
)
