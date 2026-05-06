import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] 

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
