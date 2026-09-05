import sys
from pathlib import Path

_CODE_DIR = str(Path(__file__).resolve().parents[2])
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
