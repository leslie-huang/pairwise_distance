from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pairwise_distance",
    version="0.0.1",
    author="Leslie Huang",
    author_email="lesliehuang@nyu.edu",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leslie-huang/distinctiveness_with_embeddings",
    packages=find_packages(),
    classifiers=["Programming Language :: Python :: 3"],
)
