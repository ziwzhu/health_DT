"""Pin the working directory so relative paths resolve from the project root."""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)
