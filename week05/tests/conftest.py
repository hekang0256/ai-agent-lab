import os
import sys
from pathlib import Path

os.environ["LLM_USE_MOCK"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
