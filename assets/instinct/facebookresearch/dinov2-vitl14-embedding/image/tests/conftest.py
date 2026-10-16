# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import importlib.util
import sys
from pathlib import Path

# The repo-level pytest run collects every AIM's image/tests in one session, and other
# AIMs ship modules with the same names (harness.py, service.py, config.py). So src/
# stays off sys.path and each module is loaded from its file as dinov2_<name>. The plain
# names exist only while the modules execute, for their sibling imports
# (harness.py does `from inference import ...`, as it does inside the image).
IMAGE_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
SRC_MODULES = ("golden_fixtures", "inference", "harness")


def load_src_modules(names: tuple[str, ...]) -> None:
    previous = {name: sys.modules.get(name) for name in names}
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(f"dinov2_{name}", IMAGE_SRC_DIR / f"{name}.py")
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            sys.modules[name] = module
            spec.loader.exec_module(module)
    finally:
        for name, saved in previous.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved


load_src_modules(SRC_MODULES)
