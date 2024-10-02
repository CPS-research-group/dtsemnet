from setuptools import find_packages, setup

setup(
    name='dtregnet',
    version="1.0",
    author="XXX",
    author_email="XXX@XXX",
    description="DTRegNet: DTSemNet with Regression",
    long_description=
    "Use DTSemNet with Regression",
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
    
    python_requires=">=3.8",
    package_dir={"icct": "icct",
                 "dtregnet": "dtregnet"},
    packages=find_packages('icct') + find_packages('dtregnet'),
    install_requires=['torch', 'numpy']
)


