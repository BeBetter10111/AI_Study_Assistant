CORRECTNESS_WEIGHT = 0.6
SPEED_WEIGHT = 0.3
HINT_PENALTY_WEIGHT = 0.1

# Ngưỡng thời gian phản hồi (giây): trả lời trong khoảng này trở xuống được
# tính là "nhanh" và nhận trọn điểm tốc độ; chậm hơn thì điểm tốc độ giảm dần
# về 0 (không âm).
FAST_RESPONSE_THRESHOLD_S = 30.0

# Số lượt gợi ý tối đa được tính vào phạt điểm — tránh trường hợp biên khi
# học sinh bấm gợi ý rất nhiều lần khiến điểm phạt vượt quá trọng số của nó.
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

    # Kẹp kết quả trong [0, 1] để tránh sai số dấu phẩy động đẩy giá trị ra
    # ngoài biên khi các trọng số được tinh chỉnh sau này.
    return max(0.0, min(1.0, reward))
