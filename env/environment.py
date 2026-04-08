"""Content Moderation Environment — core environment logic.

Implements the OpenEnv-compatible interface:
  - reset()  → Observation
  - step(action) → (Observation, Reward, bool, dict)
  - state()  → EnvironmentState
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from env.data import DATASET
from env.graders import ClassificationGrader, ModerationGrader, ViolationGrader
from env.models import (
    Action,
    ActionType,
    ClassificationLabel,
    EnvironmentState,
    HistoryEntry,
    ModerationDecision,
    Observation,
    Reward,
    TaskName,
    ViolationType,
)
from env.tasks import TASK_REGISTRY, TaskConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_STEPS_PER_EPISODE = 50  # generous ceiling; episodes usually end in ~20


# ---------------------------------------------------------------------------
# Reward helpers (per-step)
# ---------------------------------------------------------------------------

def _classification_reward(predicted: str, expected: str) -> Reward:
    """Compute step reward for the classification task."""
    if predicted == expected:
        return Reward(value=1.0, reason="Correct classification.")
    # Borderline partial credit
    borderline = {
        frozenset({"spam", "unsafe"}),
        frozenset({"safe", "spam"}),
    }
    if frozenset({predicted, expected}) in borderline:
        return Reward(value=0.5, reason="Partially correct — borderline item.")
    return Reward(value=-0.3, reason=f"Wrong classification: predicted '{predicted}', expected '{expected}'.")


def _violation_reward(predicted: str, expected: str) -> Reward:
    """Compute step reward for the violation detection task."""
    if predicted == expected:
        return Reward(value=1.0, reason="Correct violation detection.")
    if expected != "none" and predicted != "none":
        return Reward(
            value=0.5,
            reason=f"Flagged a violation but wrong type: predicted '{predicted}', expected '{expected}'.",
        )
    return Reward(value=-0.4, reason=f"Wrong violation type: predicted '{predicted}', expected '{expected}'.")


def _moderation_reward(
    predicted: str,
    expected: str,
    reason_text: Optional[str] = None,
) -> Reward:
    """Compute step reward for the moderation decision task."""
    if predicted == expected:
        # Bonus for meaningful reasoning keywords
        bonus = 0.0
        if reason_text:
            keywords = {"policy", "harmful", "borderline", "violat", "safe", "scam", "spam", "escalat", "abuse", "phish"}
            matches = sum(1 for kw in keywords if kw in reason_text.lower())
            if matches >= 2:
                bonus = 0.5
        return Reward(
            value=1.0 + bonus,
            reason="Correct moderation decision." + (f" Reasoning bonus +{bonus:.1f}." if bonus else ""),
        )
    if predicted == "approve" and expected in ("reject", "escalate"):
        return Reward(
            value=-0.5,
            reason=f"Dangerous miss: approved content that should be '{expected}'.",
        )
    if predicted == "escalate" and expected == "reject":
        return Reward(value=-0.3, reason="Escalated instead of rejecting — cautious but wrong.")
    if predicted == "reject" and expected == "escalate":
        return Reward(value=-0.3, reason="Rejected instead of escalating — too aggressive.")
    return Reward(value=-0.3, reason=f"Wrong moderation decision: predicted '{predicted}', expected '{expected}'.")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ContentModerationEnv:
    """OpenEnv-compatible Content Moderation Environment."""

    def __init__(self, task_name: str = "classification"):
        if task_name not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task '{task_name}'. Choose from: {list(TASK_REGISTRY.keys())}"
            )
        self._task_config: TaskConfig = TASK_REGISTRY[task_name]
        self._items = list(DATASET)  # deterministic copy
        self._current_index: int = 0
        self._step_count: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._history: List[HistoryEntry] = []
        self._last_action_error: Optional[str] = None
        self._results: List[Tuple[str, str]] = []  # (predicted, expected)
        self._reasons: List[Optional[str]] = []

    # ---- OpenEnv API -------------------------------------------------------

    def reset(self) -> Observation:
        """Initialize or re-initialize the episode and return the first observation."""
        self._current_index = 0
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._history = []
        self._last_action_error = None
        self._results = []
        self._reasons = []
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Execute one action and return (observation, reward, done, info)."""
        if self._done:
            obs = self._make_observation()
            return (
                obs,
                Reward(value=0.0, reason="Episode already finished."),
                True,
                self._make_info(valid_action=False),
            )

        self._step_count += 1

        # --- Validate action ---
        error = self._validate_action(action)
        if error is not None:
            self._last_action_error = error
            reward = Reward(value=-0.1, reason=f"Invalid action: {error}")
            self._cumulative_reward += reward.value
            # Check max steps
            if self._step_count >= MAX_STEPS_PER_EPISODE:
                self._done = True
            obs = self._make_observation()
            return obs, reward, self._done, self._make_info(valid_action=False)

        self._last_action_error = None

        # --- Extract predicted value ---
        predicted = self._extract_action_value(action)
        item = self._items[self._current_index]
        expected = self._task_config.ground_truth_accessor(item)

        # --- Compute reward ---
        reward = self._compute_reward(predicted, expected, action.reason)
        self._cumulative_reward += reward.value

        # --- Record history ---
        entry = HistoryEntry(
            step=self._step_count,
            content_id=item.id,
            action_type=action.action_type.value,
            action_value=predicted,
            reward=reward.value,
        )
        self._history.append(entry)
        self._results.append((predicted, expected))
        self._reasons.append(action.reason)

        # --- Advance to next item ---
        self._current_index += 1
        if self._current_index >= len(self._items) or self._step_count >= MAX_STEPS_PER_EPISODE:
            self._done = True

        obs = self._make_observation()
        info = self._make_info(valid_action=True)

        return obs, reward, self._done, info

    def state(self) -> EnvironmentState:
        """Return the full internal state of the environment."""
        return EnvironmentState(
            task_name=self._task_config.name.value,
            current_item_index=self._current_index,
            total_items=len(self._items),
            step_count=self._step_count,
            max_steps=MAX_STEPS_PER_EPISODE,
            done=self._done,
            cumulative_reward=self._cumulative_reward,
            history=list(self._history),
            last_action_error=self._last_action_error,
        )

    # ---- Internal helpers --------------------------------------------------

    def _make_observation(self) -> Observation:
        """Build the current observation for the agent."""
        if self._current_index < len(self._items):
            item = self._items[self._current_index]
            content_id = item.id
            content_text = item.text
        else:
            content_id = "none"
            content_text = "No more items to review."

        remaining = max(0, len(self._items) - self._current_index)

        return Observation(
            task_name=self._task_config.name.value,
            content_id=content_id,
            content_text=content_text,
            policy_context=self._task_config.policy_context,
            available_actions=self._task_config.available_actions,
            step_count=self._step_count,
            remaining_items=remaining,
            history=list(self._history),
            last_action_error=self._last_action_error,
        )

    def _validate_action(self, action: Action) -> Optional[str]:
        """Return an error string if the action is invalid, else None."""
        expected_type = self._task_config.action_type

        if action.action_type != expected_type:
            return (
                f"Expected action_type '{expected_type.value}' for task "
                f"'{self._task_config.name.value}', got '{action.action_type.value}'."
            )

        if action.action_type == ActionType.CLASSIFY:
            if action.label is None:
                return "action_type 'classify' requires a 'label' field."
            if action.label.value not in self._task_config.available_actions:
                return f"Invalid label '{action.label.value}'. Choose from: {self._task_config.available_actions}."

        elif action.action_type == ActionType.FLAG:
            if action.violation_type is None:
                return "action_type 'flag' requires a 'violation_type' field."
            if action.violation_type.value not in self._task_config.available_actions:
                return f"Invalid violation_type '{action.violation_type.value}'. Choose from: {self._task_config.available_actions}."

        elif action.action_type == ActionType.ROUTE:
            if action.decision is None:
                return "action_type 'route' requires a 'decision' field."
            if action.decision.value not in self._task_config.available_actions:
                return f"Invalid decision '{action.decision.value}'. Choose from: {self._task_config.available_actions}."

        return None

    def _extract_action_value(self, action: Action) -> str:
        """Return the string value of the agent's choice."""
        if action.action_type == ActionType.CLASSIFY:
            return action.label.value  # type: ignore[union-attr]
        if action.action_type == ActionType.FLAG:
            return action.violation_type.value  # type: ignore[union-attr]
        if action.action_type == ActionType.ROUTE:
            return action.decision.value  # type: ignore[union-attr]
        raise ValueError(f"Unknown action_type: {action.action_type}")

    def _compute_reward(
        self,
        predicted: str,
        expected: str,
        reason_text: Optional[str] = None,
    ) -> Reward:
        """Dispatch to the task-specific reward function."""
        task = self._task_config.name
        if task == TaskName.CLASSIFICATION:
            return _classification_reward(predicted, expected)
        if task == TaskName.VIOLATION_DETECTION:
            return _violation_reward(predicted, expected)
        if task == TaskName.MODERATION_DECISION:
            return _moderation_reward(predicted, expected, reason_text)
        raise ValueError(f"Unknown task: {task}")

    def _make_info(self, valid_action: bool) -> Dict[str, Any]:
        """Build the info dict returned alongside step results."""
        info: Dict[str, Any] = {
            "valid_action": valid_action,
            "expected_format": f"action_type='{self._task_config.action_type.value}' with one of {self._task_config.available_actions}",
            "last_action_error": self._last_action_error,
        }
        if self._done:
            info["task_score"] = self._compute_task_score()
        return info

    def _compute_task_score(self) -> float:
        """Use the appropriate grader to compute a final task score in [0, 1]."""
        task = self._task_config.name
        if task == TaskName.CLASSIFICATION:
            return ClassificationGrader().grade(self._results)
        if task == TaskName.VIOLATION_DETECTION:
            return ViolationGrader().grade(self._results)
        if task == TaskName.MODERATION_DECISION:
            return ModerationGrader().grade(self._results, self._reasons)
        return 0.0
