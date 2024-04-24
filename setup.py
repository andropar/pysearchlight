from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="pysearchlight",
    version="1.0.0",
    description="A simple and customizable implementation of the searchlight analysis method.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andropar/pysearchlight",
    author="Johannes Roth",
    author_email="johannes@roth24.de",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="searchlight, neuroscience, analysis, rsa, decoding",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7, <4",
    install_requires=["numpy", "joblib", "numba", "tqdm" "scikit-learn"],
)
