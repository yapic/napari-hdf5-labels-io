try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._reader import h5_to_napari
from ._writer import project_to_h5


