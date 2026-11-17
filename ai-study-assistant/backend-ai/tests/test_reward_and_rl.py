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


@pytest.fixture
def memory_state(monkeypatch):
    """In-memory replacement for the Redis/DB state store (update_state is async and takes a db session)."""
    store: dict = {}

    async def load(user_id, db):
        return dict(store.get(user_id, {}))

    async def save(user_id, state, db):
        store[user_id] = dict(state)

    monkeypatch.setattr(agent, "load_state", load)
    monkeypatch.setattr(agent, "save_state", save)
    return store


async def test_difficulty_promotes_after_consecutive_correct_answers(memory_state):
    user_id = "test-user-promote"
    difficulty = "easy"
    for _ in range(5):
        difficulty = await agent.update_state(
            user_id, {"is_correct": True, "response_time_s": 1.0, "hints_used": 0}, db=None
        )
    assert difficulty == "hard"


async def test_difficulty_demotes_after_consecutive_wrong_answers(memory_state):
    user_id = "test-user-demote"
    memory_state[user_id] = {"difficulty_index": 1, "reward_history": []}  # start at "medium"
    difficulty = "medium"
    for _ in range(5):
        difficulty = await agent.update_state(
            user_id, {"is_correct": False, "response_time_s": 40.0, "hints_used": 3}, db=None
        )
    assert difficulty == "easy"
