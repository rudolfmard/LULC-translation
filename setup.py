import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mmt",
    version="0.1.1",
    author="Luc Baudoux (original idea), Thomas Rieutord (dev > 0.1.0)",
    author_email="thomas.rieutord@met.ie",
    description="""Multi land-use/land-cover translation network.
    Fork from original repository: 31 Aug 2023""",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ThomasRieutord/MT-MLULC",
    packages=setuptools.find_packages(),
    classifiers=(
        "Environment :: Console"
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
        "Development Status :: 2 - Pre-Alpha",
    ),
)
