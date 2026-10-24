"""Cập nhật độ khó (RL State) dựa trên phong độ người học."""
import logging
import threading
from typing import Dict, List

from app.rl.reward import compute_reward

logger = logging.getLogger(__name__)

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
PROMOTE_THRESHOLD = 0.7
DEMOTE_THRESHOLD = 0.35

# Trạng thái RL theo user_id. Trước đây update_state() gọi thẳng
# update_state_in_db(...) — một hàm chưa từng được định nghĩa ở đâu trong
# code base, nên endpoint /quiz/{id}/submit sẽ luôn crash với NameError.
# Ở đây dùng bộ nhớ tiến trình (thread-safe) làm nơi lưu trạng thái tạm thời.
# TODO(sản xuất): thay bằng bảng riêng trong Postgres/Redis để trạng thái
# không mất khi restart service hoặc khi chạy nhiều instance backend.
_state_lock = threading.Lock()
_user_states: Dict[str, Dict] = {}


def _get_user_state(user_id: str) -> Dict:
    return _user_states.setdefault(user_id, {"difficulty_index": 0, "reward_history": []})


def update_state(user_id: str, quiz_result: dict) -> str:
    """Nhận kết quả làm bài, cập nhật State và chọn độ khó câu hỏi tiếp theo.

    :param user_id: định danh người dùng
    :param quiz_result: dict gồm is_correct (bool), response_time_s (float),
        hints_used (int)
    :return: nhãn độ khó tiếp theo — "easy" | "medium" | "hard"
    """
    if not user_id:
        raise ValueError("user_id not provided.")

    is_correct = bool(quiz_result.get("is_correct", False))
    response_time_s = float(quiz_result.get("response_time_s", 0.0))
    hints_used = int(quiz_result.get("hints_used", 0))

    reward = compute_reward(is_correct, response_time_s, hints_used)

    with _state_lock:
        state = _get_user_state(user_id)
        history: List[float] = state["reward_history"]
        history.append(reward)
        if len(history) > 200:
            del history[: len(history) - 200]

        if reward >= PROMOTE_THRESHOLD:
            state["difficulty_index"] = min(state["difficulty_index"] + 1, len(DIFFICULTY_LEVELS) - 1)
        elif reward <= DEMOTE_THRESHOLD:
            state["difficulty_index"] = max(state["difficulty_index"] - 1, 0)

        next_difficulty = DIFFICULTY_LEVELS[state["difficulty_index"]]

    logger.info(f"RL update user={user_id} reward={reward:.2f} next_difficulty={next_difficulty}")

    return next_difficulty
