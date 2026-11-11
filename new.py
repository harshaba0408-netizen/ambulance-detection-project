"""Legacy launch command for the ambulance dashboard.

The implementation now lives in `src/ambulance_system/dashboard/`. This file
stays as a tiny entry point so the documented `python new.py` workflow keeps
working without installing the package.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ambulance_system.dashboard.app import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
