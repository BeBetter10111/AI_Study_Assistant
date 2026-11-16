CORRECTNESS_WEIGHT = 0.6
SPEED_WEIGHT = 0.3
HINT_PENALTY_WEIGHT = 0.1

# Response-time threshold (seconds): answers within this time count as "fast"
# and receive the full speed score; slower answers get a gradually decreasing
# speed score down to 0 (never negative).
FAST_RESPONSE_THRESHOLD_S = 30.0

# Maximum number of hints counted toward the penalty — prevents the penalty
# from exceeding its weight if a student requests hints many times.
MAX_HINTS_CONSIDERED = 5


def compute_reward(is_correct: bool, response_time_s: float, hints_used: int) -> float:
    """Tính điểm thưởng (reward) trong khoảng [0, 1] từ kết quả một lượt làm bài.

    - correctness: 1.0 nếu trả lời đúng, ngược lại 0.0.
    - speed: càng nhanh hơn FAST_RESPONSE_THRESHOLD_S thì điểm càng cao.
    - hint penalty: mỗi lần dùng gợi ý trừ dần điểm, tối đa MAX_HINTS_CONSIDERED lượt.
    """
    if response_time_s < 0:
        raise ValueError("response_time_s can not be negative.")
    if hints_used < 0:
        raise ValueError("hints_used can not be negative.")

    correctness_score = 1.0 if is_correct else 0.0

    clamped_time = min(response_time_s, FAST_RESPONSE_THRESHOLD_S)
    speed_score = 1.0 - (clamped_time / FAST_RESPONSE_THRESHOLD_S)

    clamped_hints = min(hints_used, MAX_HINTS_CONSIDERED)
    hint_penalty = clamped_hints / MAX_HINTS_CONSIDERED

    reward = (
        CORRECTNESS_WEIGHT * correctness_score
        + SPEED_WEIGHT * speed_score
        - HINT_PENALTY_WEIGHT * hint_penalty
    )

    # Clamp the result to [0, 1] so floating-point errors do not push it
    # out of range if the weights are tuned later.
    return max(0.0, min(1.0, reward))
