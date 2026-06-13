# Tells tests to add the project root to the Python path so 'src' is importable.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))