# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Puts the anemoi-models/anemoi-graphs folder of ``AIFS_VARIANT`` first on the import path.

AIFS Single 2.0 needs anemoi-models 0.9.3 / anemoi-graphs 0.6.4 and AIFS ENS 2.0
needs 0.11.2 / 0.8.1; the image keeps each pair in ``<AIFS_ANEMOI_ROOT>/<variant>``.
``activate()`` must run before anything imports anemoi.models, so one process
serves one variant. ``python -m anemoi_variant --check`` verifies the active one.
Standard library only.
"""

import os
import sys

VARIANTS = ("single", "ens")
DEFAULT_VARIANT = "single"


def root():
    return os.environ.get("AIFS_ANEMOI_ROOT", "").strip() or "/opt/anemoi"


def name():
    """The configured variant; raises ValueError on an unknown name."""
    value = os.environ.get("AIFS_VARIANT", "").strip().lower() or DEFAULT_VARIANT
    if value not in VARIANTS:
        raise ValueError("AIFS_VARIANT=%r; use one of %s" % (value, ", ".join(VARIANTS)))
    return value


def folder(variant=None):
    return os.path.join(root(), variant or name())


def _loaded_from():
    """Folder anemoi.models was imported from, or None if it is not imported."""
    mod = sys.modules.get("anemoi.models")
    if mod is None or not getattr(mod, "__file__", None):
        return None
    # <folder>/anemoi/models/__init__.py
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(mod.__file__))))


def activate():
    """Select the variant's folder for this process and its children; returns the variant name.

    A missing folder (a checkout without the image's /opt/anemoi) leaves the path alone.
    """
    variant = name()
    path = folder(variant)
    loaded = _loaded_from()
    if loaded is not None:
        if os.path.isdir(path) and loaded != os.path.realpath(path):
            raise RuntimeError(
                "anemoi.models is already imported from %s, but AIFS_VARIANT=%s needs %s; "
                "set AIFS_VARIANT before the process starts" % (loaded, variant, path)
            )
        return variant
    if os.path.isdir(path):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
        rest = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and p != path]
        os.environ["PYTHONPATH"] = os.pathsep.join([path] + rest)
    return variant


def check():
    """Fail unless the active folder's requirements are met and anemoi imports from it."""
    from importlib import metadata

    from packaging.requirements import Requirement

    variant = activate()
    path = folder(variant)
    if not os.path.isdir(path):
        raise SystemExit("no folder %s for AIFS_VARIANT=%s" % (path, variant))
    problems = []
    dists = list(metadata.distributions(path=[path]))
    if not dists:
        problems.append("no packages in %s" % path)
    for dist in dists:
        for text in dist.requires or []:
            req = Requirement(text)
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            try:
                have = metadata.version(req.name)
            except metadata.PackageNotFoundError:
                problems.append("%s needs %s: not installed" % (dist.metadata["Name"], req))
                continue
            if not req.specifier.contains(have, prereleases=True):
                problems.append("%s needs %s: have %s" % (dist.metadata["Name"], req, have))

    import anemoi.graphs
    import anemoi.models

    for mod in (anemoi.models, anemoi.graphs):
        if not os.path.realpath(mod.__file__).startswith(os.path.realpath(path) + os.sep):
            problems.append("%s imported from %s, not %s" % (mod.__name__, mod.__file__, path))
    if problems:
        raise SystemExit("AIFS_VARIANT=%s: %s" % (variant, "; ".join(problems)))
    print(
        "AIFS_VARIANT=%s folder=%s anemoi.models.__version__=%s anemoi.graphs.__version__=%s packages=%s"
        % (
            variant,
            path,
            anemoi.models.__version__,
            anemoi.graphs.__version__,
            ",".join(sorted("%s==%s" % (d.metadata["Name"], d.version) for d in dists)),
        )
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        check()
    else:
        print(activate())
