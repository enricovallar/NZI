from setuptools import setup, find_packages

# Read version from a file or define it here
VERSION = "0.1.0"  # Update this as needed

setup(
    name="nzi-phc-finder",
    version=VERSION,
    description="This package is a collection of scripts to find the photonic crystal nzi modes",
    author="Enrico Vallar",
    author_email="enrico.vallar2000@gmail.com",  # Add your email
    url="https://github.com/enricovallar/nzi-lithium-niobate",
    packages=find_packages(),
    python_requires=">=3.11",
    license="GPL-3.0-or-later",
    license_file="LICENSE",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3.11",
    ],

)