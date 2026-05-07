import sys
from pathlib import Path

# 1. 自動定位專案根目錄
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] 

# 2. 路徑注入
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 這裡可以放一些全域變數，例如版本號
__version__ = "1.0.0"