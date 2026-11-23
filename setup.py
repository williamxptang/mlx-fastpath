from pathlib import Path

from setuptools import setup

# VERSION is the one source of the version: first line = live version.
setup(version=(Path(__file__).parent / "VERSION").read_text().splitlines()[0].strip())
