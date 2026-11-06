import os
import sys
from pathlib import Path

# 所有测试离线跑 Mock；把 week04 根目录加入 sys.path，便于 import llm/apps
os.environ["LLM_USE_MOCK"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
