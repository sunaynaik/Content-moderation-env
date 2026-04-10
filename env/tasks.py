"""Task definitions for the Content Moderation Environment.

Each task defines:
  - action_type expected from the agent
  - available action values
  - policy context shown to the agent
  - ground-truth accessor for grading
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from env.models import (
    ActionType,
    ClassificationLabel,
    ContentItem,
    ModerationDecision,
    TaskName,
    ViolationType,
)


@dataclass(frozen=True)
class TaskConfig:
    """Configuration for a single task."""

    name: TaskName
    action_type: ActionType
    available_actions: List[str]
    policy_context: str
    ground_truth_accessor: Callable[[ContentItem], str]


def _get_classification_label(item: ContentItem) -> str:
    return item.classification_label.value


def _get_violation_label(item: ContentItem) -> str:
    return item.violation_label.value


def _get_moderation_decision(item: ContentItem) -> str:
    return item.moderation_decision.value


# ---------------------------------------------------------------------------
# Policy context strings
# ---------------------------------------------------------------------------

CLASSIFICATION_POLICY = (
    "You are a content moderator. Classify the following content into one of "
    "three categories:\n"
    "  - safe: normal, harmless content with no policy issues\n"
    "  - spam: promotional, repetitive, off-topic, or deceptive advertising\n"
    "  - unsafe: abusive, hateful, threatening, or clearly harmful content\n"
    "Choose the single best label."
)

VIOLATION_POLICY = (
    "You are a policy violation detector. Identify the specific type of "
    "violation in the following content:\n"
    "  - harassment: insulting, degrading, or targeted personal abuse\n"
    "  - hate_speech: discriminatory or group-targeting language\n"
    "  - scam: fraudulent offers, phishing, or deceptive claims\n"
    "  - none: no policy violation detected\n"
    "Be precise — choose the most specific violation type."
)

MODERATION_POLICY = (
    "You are a senior content moderator making the final routing decision. "
    "Decide what should happen to the following content:\n"
    "  - approve: content is safe and can be published\n"
    "  - reject: content clearly violates platform policy\n"
    "  - escalate: content is borderline, ambiguous, or policy-sensitive "
    "and should be reviewed by a human moderator\n"
    "Consider edge cases carefully. Provide a brief reason for your "
    "decision if possible."
)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, TaskConfig] = {
    TaskName.CLASSIFICATION.value: TaskConfig(
        name=TaskName.CLASSIFICATION,
        action_type=ActionType.CLASSIFY,
        available_actions=[label.value for label in ClassificationLabel],
        policy_context=CLASSIFICATION_POLICY,
        ground_truth_accessor=_get_classification_label,
    ),
    TaskName.VIOLATION_DETECTION.value: TaskConfig(
        name=TaskName.VIOLATION_DETECTION,
        action_type=ActionType.FLAG,
        available_actions=[vt.value for vt in ViolationType],
        policy_context=VIOLATION_POLICY,
        ground_truth_accessor=_get_violation_label,
    ),
    TaskName.MODERATION_DECISION.value: TaskConfig(
        name=TaskName.MODERATION_DECISION,
        action_type=ActionType.ROUTE,
        available_actions=[md.value for md in ModerationDecision] + ["investigate"],
        policy_context=MODERATION_POLICY,
        ground_truth_accessor=_get_moderation_decision,
    ),
}
