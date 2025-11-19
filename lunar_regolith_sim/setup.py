from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lunar-regolith-sim",
    version="0.1.0",
    author="Lunar Regolith Research",
    description="Simulation of regolith flow on lunar slopes to model elephant hide textures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lunar-regolith-sim",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pillow>=8.3.0",
        "numba>=0.54.0",
    ],
)
