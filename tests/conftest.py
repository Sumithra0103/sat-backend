"""
Pytest configuration and global sys.path fixture for SatQuery AI.
"""

import sys
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) in sys.path:
    sys.path.remove(str(_ROOT_DIR))
sys.path.insert(0, str(_ROOT_DIR))
