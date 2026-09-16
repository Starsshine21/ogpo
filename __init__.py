"""Compatibility package for legacy ``dexjoco.ogpo`` imports.

The installable package lives in ``src/ogpo``.  Extending ``__path__`` lets
historical scripts resolve submodules such as ``dexjoco.ogpo.trainer`` without
maintaining a second copy of the code.
"""

from pathlib import Path


_SOURCE_PACKAGE = Path(__file__).resolve().parent / "src" / "ogpo"
if _SOURCE_PACKAGE.is_dir():
    __path__.insert(0, str(_SOURCE_PACKAGE))  # type: ignore[name-defined]
