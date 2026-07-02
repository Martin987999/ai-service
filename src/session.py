"""In-memory multi-turn session store.

多轮会话记忆:按 session_id 保存最近若干轮 (user, assistant)。
带 TTL 与容量上限;生产可替换为 Redis 等共享存储(接口不变)。
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str   # "user" | "assistant"
    content: str


@dataclass
class _Session:
    turns: list[Turn] = field(default_factory=list)
    ts: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, max_turns: int = 12, ttl_s: int = 3600, max_sessions: int = 2000):
        self.max_turns = max_turns
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions
        self._s: "OrderedDict[str, _Session]" = OrderedDict()
        self._lock = threading.RLock()  # guards _s against concurrent mutation/iteration

    def new_id(self) -> str:
        return uuid.uuid4().hex[:16]

    def history(self, session_id: str) -> list[Turn]:
        with self._lock:
            self._evict()
            s = self._s.get(session_id)
            return list(s.turns) if s else []

    def append(self, session_id: str, user: str, assistant: str) -> None:
        with self._lock:
            self._evict()
            s = self._s.get(session_id)
            if s is None:
                s = _Session()
                self._s[session_id] = s
            s.turns.append(Turn("user", user))
            s.turns.append(Turn("assistant", assistant))
            s.turns = s.turns[-self.max_turns:]
            s.ts = time.time()
            self._s.move_to_end(session_id)
            while len(self._s) > self.max_sessions:
                self._s.popitem(last=False)

    def _evict(self) -> None:
        # caller must hold self._lock
        now = time.time()
        stale = [k for k, v in self._s.items() if now - v.ts > self.ttl_s]
        for k in stale:
            self._s.pop(k, None)
