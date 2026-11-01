"""Run the complete Intent2CAD data synthesis pipeline."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intent2cad.src.pipeline import main

if __name__ == "__main__":
    main()
