"""Offline smoke test — runs all three tasks with deterministic actions.

No API key or LLM required. Validates that the environment works correctly.
"""

from env.environment import ContentModerationEnv
from env.models import (
    Action,
    ActionType,
    ClassificationLabel,
    ModerationDecision,
    ViolationType,
)
from env.data import DATASET


def test_classification():
    """Test classification task end-to-end with ground-truth actions."""
    env = ContentModerationEnv(task_name="classification")
    obs = env.reset()
    assert obs.task_name == "classification"
    assert obs.remaining_items == 20
    assert obs.step_count == 0

    total_reward = 0.0
    done = False
    steps = 0

    while not done:
        # Use ground-truth label for perfect score
        item = DATASET[steps]
        action = Action(
            action_type=ActionType.CLASSIFY,
            label=item.classification_label,
        )
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        steps += 1

    assert done is True
    assert steps == 20
    assert info["valid_action"] is True
    assert "task_score" in info
    assert info["task_score"] == 1.0  # Perfect answers
    print(f"✓ classification: score={info['task_score']}, reward={total_reward:.1f}, steps={steps}")


def test_violation_detection():
    """Test violation detection task end-to-end with ground-truth actions."""
    env = ContentModerationEnv(task_name="violation_detection")
    obs = env.reset()
    assert obs.task_name == "violation_detection"

    total_reward = 0.0
    done = False
    steps = 0

    while not done:
        item = DATASET[steps]
        action = Action(
            action_type=ActionType.FLAG,
            violation_type=item.violation_label,
        )
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        steps += 1

    assert done is True
    assert steps == 20
    assert info["task_score"] == 1.0
    print(f"✓ violation_detection: score={info['task_score']}, reward={total_reward:.1f}, steps={steps}")


def test_moderation_decision():
    """Test moderation decision task end-to-end with ground-truth actions."""
    env = ContentModerationEnv(task_name="moderation_decision")
    obs = env.reset()
    assert obs.task_name == "moderation_decision"

    total_reward = 0.0
    done = False
    steps = 0

    while not done:
        item = DATASET[steps]
        action = Action(
            action_type=ActionType.ROUTE,
            decision=item.moderation_decision,
            reason="This content is borderline and requires policy review.",
        )
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        steps += 1

    assert done is True
    assert steps == 20
    assert info["task_score"] == 1.0
    print(f"✓ moderation_decision: score={info['task_score']}, reward={total_reward:.1f}, steps={steps}")


def test_invalid_action():
    """Test that invalid actions are handled gracefully."""
    env = ContentModerationEnv(task_name="classification")
    env.reset()

    # Wrong action type
    action = Action(action_type=ActionType.FLAG, violation_type=ViolationType.NONE)
    obs, reward, done, info = env.step(action)
    assert info["valid_action"] is False
    assert reward.value == -0.1
    assert obs.last_action_error is not None
    print(f"✓ invalid action handled: error='{obs.last_action_error[:50]}...'")


def test_state():
    """Test that state() returns correct internal state."""
    env = ContentModerationEnv(task_name="classification")
    env.reset()

    state = env.state()
    assert state.task_name == "classification"
    assert state.step_count == 0
    assert state.done is False
    assert state.total_items == 20
    assert state.max_steps == 50

    # Take one step
    action = Action(action_type=ActionType.CLASSIFY, label=ClassificationLabel.SAFE)
    env.step(action)

    state = env.state()
    assert state.step_count == 1
    assert state.current_item_index == 1
    assert len(state.history) == 1
    print(f"✓ state() correct: step={state.step_count}, items={state.total_items}")


def test_episode_done():
    """Test that stepping on a done environment returns gracefully."""
    env = ContentModerationEnv(task_name="classification")
    env.reset()

    # Exhaust all items
    for i in range(20):
        action = Action(action_type=ActionType.CLASSIFY, label=ClassificationLabel.SAFE)
        _, _, done, _ = env.step(action)
    assert done is True

    # Step again after done
    _, reward, done, info = env.step(
        Action(action_type=ActionType.CLASSIFY, label=ClassificationLabel.SAFE)
    )
    assert done is True
    assert reward.value == 0.0
    print("✓ post-done step handled gracefully")


if __name__ == "__main__":
    print("Running Content Moderation Environment smoke tests...\n")
    test_classification()
    test_violation_detection()
    test_moderation_decision()
    test_invalid_action()
    test_state()
    test_episode_done()
    print("\n✅ All tests passed!")
