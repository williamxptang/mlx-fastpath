"""Version: the repo's root VERSION file is the one source.
Build time: setuptools reads VERSION_FROM_FILE below. Run time: the installed package metadata,
falling back to the VERSION file when running from the source tree."""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_FILE = Path(__file__).resolve().parent.parent / "VERSION"


def _from_file():
    return _FILE.read_text().splitlines()[0].strip() if _FILE.exists() else None


VERSION_FROM_FILE = _from_file()


def _resolve():
    if VERSION_FROM_FILE is not None and (Path(__file__).resolve().parent.parent / "pyproject.toml").exists():
        return VERSION_FROM_FILE  # running from the source tree
    try:
        return version("mlx-fastpath")
    except PackageNotFoundError:
        return VERSION_FROM_FILE or "0+unknown"


__version__ = _resolve()
