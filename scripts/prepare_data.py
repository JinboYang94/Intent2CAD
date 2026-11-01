"""Download and prepare the released Intent2CAD benchmark."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intent2cad.datasets import main

if __name__ == "__main__":
    main()
