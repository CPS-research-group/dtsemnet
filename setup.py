from setuptools import find_packages, setup

setup(
    name='dtsemnet',
    version="1.0",
    author="XXX",
    author_email="XXX@XXX",
    description="DTNet: DT to NN Interconversion",
    long_description=
    "Code to convert DT to NN, train the NN and get back the DT from NN",
    long_description_content_type="text/markdown",
    url="",
    project_urls={
        "Bug Tracker": "",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "dtsemnet"},
    requires=['numpy', 'torch', 'gym', 'matplotlib', 'scipy', 'pandas'],
    packages=find_packages(where="dtsemnet"),
    python_requires=">=3.8",
)
