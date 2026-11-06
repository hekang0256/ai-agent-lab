"""记忆（Memory）：让助手跨轮次、跨会话记住用户。

分层设计：
- 短期记忆 = 当前会话的消息历史（见 ChatSession.history，由 ContextManager 管理）
- 长期记忆 = 本模块的 MemoryStore：键值形式的用户事实（姓名/偏好/预算…），JSON 文件持久化

长期记忆的使用方式：**每轮把画像文本注入 system prompt**，而不是塞进对话历史。
这样它不会被滑动窗口裁掉，也不会随对话变长线性膨胀。

解析用户显式记忆指令（"记住我叫张三" / "忘掉我的城市"）用规则完成，
不依赖 LLM，离线可测；后续可替换为 LLM 自动抽取（week13 Evaluation 可评估抽取质量）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# 内部统一 key → 展示用中文名
_KEY_ALIASES = {"名字": "name", "姓名": "name", "喜好": "likes", "偏好": "likes"}
_DISPLAY_NAMES = {"name": "姓名", "likes": "喜好"}


def _normalize_key(key: str) -> str:
    key = key.strip().strip("。")
    return _KEY_ALIASES.get(key, key)


@dataclass
class MemoryCommand:
    action: str   # remember / forget
    key: str
    value: str = ""


def parse_memory_command(text: str) -> MemoryCommand | None:
    """解析显式记忆指令，不匹配则返回 None（当作普通对话处理）。"""
    t = text.strip().rstrip("。！!")

    m = re.match(r"^记住我叫(.+)$", t)
    if m:
        return MemoryCommand("remember", "name", m.group(1).strip())
    m = re.match(r"^记住我的名字是(.+)$", t)
    if m:
        return MemoryCommand("remember", "name", m.group(1).strip())
    m = re.match(r"^记住我喜欢(.+)$", t)
    if m:
        return MemoryCommand("remember", "likes", m.group(1).strip())
    m = re.match(r"^记住我的?(.+?)是(.+)$", t)
    if m:
        return MemoryCommand("remember", _normalize_key(m.group(1)), m.group(2).strip())
    m = re.match(r"^(?:忘掉|忘记)我的?(.+)$", t)
    if m:
        return MemoryCommand("forget", _normalize_key(m.group(1)))
    return None


class MemoryStore:
    """JSON 文件持久化的长期记忆。按 user_id 隔离不同用户。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict = {"users": {}}

    @classmethod
    def load(cls, path: str | Path) -> "MemoryStore":
        store = cls(path)
        if store.path.is_file():
            store._data = json.loads(store.path.read_text(encoding="utf-8"))
            store._data.setdefault("users", {})
        return store

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _user(self, user_id: str) -> dict:
        return self._data["users"].setdefault(user_id, {"facts": {}})

    # ---------- 读写 ----------
    def remember(self, key: str, value: str, *, user_id: str = "default") -> None:
        self._user(user_id)["facts"][key] = value
        self.save()

    def recall(self, key: str, *, user_id: str = "default", default=None):
        return self._user(user_id)["facts"].get(key, default)

    def forget(self, key: str, *, user_id: str = "default") -> bool:
        facts = self._user(user_id)["facts"]
        existed = key in facts
        facts.pop(key, None)
        self.save()
        return existed

    def facts(self, *, user_id: str = "default") -> dict:
        return dict(self._user(user_id)["facts"])

    def clear(self, *, user_id: str = "default") -> None:
        self._data["users"][user_id] = {"facts": {}}
        self.save()

    # ---------- 注入 prompt ----------
    def profile_text(self, *, user_id: str = "default") -> str:
        """生成注入 system prompt 的画像文本；无记忆时返回空串。"""
        facts = self.facts(user_id=user_id)
        if not facts:
            return ""
        lines = [
            f"- {_DISPLAY_NAMES.get(k, k)}: {v}" for k, v in facts.items()
        ]
        return "# 用户画像（长期记忆，请在回答中自然地利用）\n" + "\n".join(lines)
