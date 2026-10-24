"""Test thuần logic — không cần DB/API, chạy cực nhanh trong CI."""
import pytest

from app.rl import agent
from app.rl.reward import compute_reward


def test_reward_is_high_for_fast_correct_answer():
    assert compute_reward(True, 2.0, 0) > 0.8


def test_reward_is_zero_for_slow_wrong_answer():
    assert compute_reward(False, 60.0, 5) == 0.0


def test_reward_rejects_negative_time():
    with pytest.raises(ValueError):
        compute_reward(True, -1.0, 0)


def test_reward_rejects_negative_hints():
    with pytest.raises(ValueError):
        compute_reward(True, 1.0, -1)


def test_reward_is_always_clamped_between_0_and_1():
    assert 0.0 <= compute_reward(True, 0.0, 0) <= 1.0
    assert 0.0 <= compute_reward(False, 1000.0, 100) <= 1.0


def test_difficulty_promotes_after_consecutive_correct_answers():
    user_id = "test-user-promote"
    difficulty = "easy"
    for _ in range(5):
        difficulty = agent.update_state(
            user_id, {"is_correct": True, "response_time_s": 1.0, "hints_used": 0}
        )
    assert difficulty == "hard"


def test_difficulty_demotes_after_consecutive_wrong_answers():
    user_id = "test-user-demote"
    difficulty = "medium"
    for _ in range(5):
        difficulty = agent.update_state(
            user_id, {"is_correct": False, "response_time_s": 40.0, "hints_used": 3}
        )
    assert difficulty == "easy"
