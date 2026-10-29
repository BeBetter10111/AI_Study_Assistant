"""RL Agent — cập nhật độ khó dựa trên phong độ người học.

Thay đổi so với phiên bản cũ:
- update_state() nay là async, nhận thêm db session.
- State được load/save qua state_store (Redis + PostgreSQL) thay vì dict RAM.
- get_current_difficulty() cho phép /quiz/generate đọc difficulty hiện tại
  của user mà không cần phải update state.
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.rl.reward import compute_reward
from app.rl.state_store import load_state, save_state

logger = logging.getLogger(__name__)

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
PROMOTE_THRESHOLD = 0.7
DEMOTE_THRESHOLD = 0.35


async def get_current_difficulty(user_id: str, db: AsyncSession) -> str:
    """Trả về nhãn độ khó hiện tại của user mà không thay đổi state.

    Dùng bởi /quiz/generate để lấy difficulty phù hợp khi sinh đề.
    """
    state = await load_state(user_id, db)
    idx = max(0, min(state.get("difficulty_index", 0), len(DIFFICULTY_LEVELS) - 1))
    return DIFFICULTY_LEVELS[idx]


async def update_state(user_id: str, quiz_result: dict[str, Any], db: AsyncSession) -> str:
    """Nhận kết quả làm bài, cập nhật State và trả về độ khó tiếp theo.

    :param user_id:    định danh người dùng (UUID string)
    :param quiz_result: dict gồm is_correct (bool), response_time_s (float),
                        hints_used (int)
    :param db:         async DB session (để persist state)
    :return: nhãn độ khó tiếp theo — "easy" | "medium" | "hard"
    """
    if not user_id:
        raise ValueError("user_id không được để trống.")

    is_correct = bool(quiz_result.get("is_correct", False))
    response_time_s = float(quiz_result.get("response_time_s", 0.0))
    hints_used = int(quiz_result.get("hints_used", 0))

    reward = compute_reward(is_correct, response_time_s, hints_used)

    # Load state hiện tại
    state = await load_state(user_id, db)
    history: list[float] = state.get("reward_history", [])
    history.append(reward)

    # Cập nhật difficulty_index theo ngưỡng
    idx: int = state.get("difficulty_index", 0)
    if reward >= PROMOTE_THRESHOLD:
        idx = min(idx + 1, len(DIFFICULTY_LEVELS) - 1)
    elif reward <= DEMOTE_THRESHOLD:
        idx = max(idx - 1, 0)

    new_state = {"difficulty_index": idx, "reward_history": history}

    # Persist state (Redis + DB)
    await save_state(user_id, new_state, db)

    next_difficulty = DIFFICULTY_LEVELS[idx]
    logger.info(
        "RL update user=%s reward=%.2f next_difficulty=%s",
        user_id, reward, next_difficulty,
    )
    return next_difficulty
