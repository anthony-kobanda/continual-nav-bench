import os

from setuptools import find_packages, setup

############
# Metadata #
############

name = "offbench"
version = "0.0.1"
description = "Continual Offline Reinforcement Learning Benchmarks Library"

try:
    CURRENT_FOLDER_PATH = os.path.dirname(os.path.realpath(__file__))
    with open(f"{CURRENT_FOLDER_PATH}/../README.md","r",encoding="utf-8") as readme_file:
        long_description = readme_file.read()
except FileNotFoundError: 
    long_description = description

long_description_content_type = "text/markdown"

try:
    CURRENT_FOLDER_PATH = os.path.dirname(os.path.realpath(__file__))
    with open(f"{CURRENT_FOLDER_PATH}/../LICENSE","r",encoding="utf-8") as license_file:
        license_content = license_file.read()
except FileNotFoundError:
    license_content = description

keywords = [
    "Benchmarks",
    "Benchmarking", 
    "Benchmarking Library",
    "Continual Learning",
    "Offline Continual Learning",
    "Offline Reinforcement Learning",
    "Offline Reinforcement Learning Benchmarks",
    "Reinforcement Learning",
]

################
# REQUIREMENTS #
################

python_requires:str = ">=3.10"

try:
    with open("requirements.txt") as requirements_file:
        requirements:list[str] = requirements_file.read().splitlines()
except FileNotFoundError:
    requirements:list[str] = []

#########
# SETUP #
#########

setup(
    name=name,
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    license=license_content,
    packages=find_packages(),
    install_requires=requirements,
    python_requires=python_requires,
    keywords=keywords,
)
