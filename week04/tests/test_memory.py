"""长期记忆与记忆指令解析测试。"""
from llm.memory import MemoryStore, parse_memory_command


def test_parse_name_command():
    cmd = parse_memory_command("记住我叫何康")
    assert cmd.action == "remember" and cmd.key == "name" and cmd.value == "何康"


def test_parse_name_command_variant():
    cmd = parse_memory_command("记住我的名字是何康。")
    assert cmd.key == "name" and cmd.value == "何康"


def test_parse_likes_command():
    cmd = parse_memory_command("记住我喜欢简洁的回答")
    assert cmd.key == "likes" and cmd.value == "简洁的回答"


def test_parse_generic_command():
    cmd = parse_memory_command("记住我的城市是上海")
    assert cmd.action == "remember" and cmd.key == "城市" and cmd.value == "上海"


def test_parse_forget_command_with_alias():
    cmd = parse_memory_command("忘掉我的名字")
    assert cmd.action == "forget" and cmd.key == "name"


def test_parse_non_command_returns_none():
    assert parse_memory_command("今天天气怎么样？") is None


def test_remember_recall_forget(tmp_path):
    store = MemoryStore.load(tmp_path / "mem.json")
    store.remember("name", "何康")
    assert store.recall("name") == "何康"
    assert store.forget("name") is True
    assert store.recall("name") is None
    assert store.forget("name") is False  # 已不存在


def test_persistence_across_load(tmp_path):
    path = tmp_path / "mem.json"
    store = MemoryStore.load(path)
    store.remember("城市", "上海")
    # 模拟新进程
    reloaded = MemoryStore.load(path)
    assert reloaded.recall("城市") == "上海"


def test_user_isolation(tmp_path):
    store = MemoryStore.load(tmp_path / "mem.json")
    store.remember("name", "何康", user_id="u1")
    store.remember("name", "张三", user_id="u2")
    assert store.recall("name", user_id="u1") == "何康"
    assert store.recall("name", user_id="u2") == "张三"


def test_profile_text_injection(tmp_path):
    store = MemoryStore.load(tmp_path / "mem.json")
    assert store.profile_text() == ""  # 无记忆时不污染 system
    store.remember("name", "何康")
    text = store.profile_text()
    assert "用户画像" in text and "姓名: 何康" in text


def test_clear(tmp_path):
    store = MemoryStore.load(tmp_path / "mem.json")
    store.remember("name", "何康")
    store.clear()
    assert store.facts() == {}
