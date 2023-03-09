__all__ = ["__version__"]

import sys
from importlib.util import find_spec

if sys.version_info >= (3, 8):
    from importlib.metadata import (
        PackageNotFoundError,
        version,
    )
else:
    from importlib_metadata import (
        PackageNotFoundError,
        version,
    )

if not (find_spec("neptune") or find_spec("neptune-client")):
    msg = """
            neptune package not found.

            Install neptune package by running
                `pip install neptune`
                    """
    raise PackageNotFoundError(msg)

try:
    __version__ = version("neptune-tensorflow-keras")
except PackageNotFoundError:
    # package is not installed
    pass
