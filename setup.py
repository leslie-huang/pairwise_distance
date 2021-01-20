from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pairwisedistance",
    version="0.0.1",
    author="Leslie Huang",
    author_email="lesliehuang@nyu.edu",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leslie-huang/pairwise_distance",
    packages=find_packages(),
    classifiers=["Programming Language :: Python :: 3"],
)
