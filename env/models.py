"""Pydantic models for the Content Moderation Environment.

Defines typed Observation, Action, Reward, ContentItem, and EnvironmentState
models used throughout the environment, grading, and inference pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ClassificationLabel(str, Enum):
    SAFE = "safe"
    SPAM = "spam"
    UNSAFE = "unsafe"


class ViolationType(str, Enum):
    HARASSMENT = "harassment"
    HATE_SPEECH = "hate_speech"
    SCAM = "scam"
    NONE = "none"


class ModerationDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"


class ActionType(str, Enum):
    CLASSIFY = "classify"
    FLAG = "flag"
    ROUTE = "route"


class TaskName(str, Enum):
    CLASSIFICATION = "classification"
    VIOLATION_DETECTION = "violation_detection"
    MODERATION_DECISION = "moderation_decision"


# ---------------------------------------------------------------------------
# Content Item (dataset row)
# ---------------------------------------------------------------------------

class ContentItem(BaseModel):
    """A single piece of user-generated content with ground-truth labels."""

    id: str = Field(..., description="Unique content identifier")
    text: str = Field(..., description="The raw content text")
    classification_label: ClassificationLabel
    violation_label: ViolationType
    moderation_decision: ModerationDecision
    policy_reason: Optional[str] = Field(
        None,
        description="Human-readable policy reason for the ground-truth labels",
    )


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """Agent action submitted via step().

    Exactly one action branch should be filled depending on action_type:
      - classify  → label
      - flag      → violation_type
      - route     → decision (+ optional reason)
    """

    action_type: ActionType
    label: Optional[ClassificationLabel] = None
    violation_type: Optional[ViolationType] = None
    decision: Optional[ModerationDecision] = None
    reason: Optional[str] = Field(
        None,
        description="Optional reasoning for the moderation decision",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class HistoryEntry(BaseModel):
    """One previous step in the episode."""

    step: int
    content_id: str
    action_type: str
    action_value: str
    reward: float


class Observation(BaseModel):
    """What the agent sees after reset() or step()."""

    task_name: str
    content_id: str
    content_text: str
    policy_context: str = Field(
        ...,
        description="Policy instructions relevant to the current task",
    )
    available_actions: List[str]
    step_count: int
    remaining_items: int
    history: List[HistoryEntry] = Field(default_factory=list)
    last_action_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Structured reward returned by the environment."""

    value: float = Field(..., description="Numeric reward for this step")
    reason: str = Field(
        ...,
        description="Human-readable explanation of the reward",
    )


# ---------------------------------------------------------------------------
# Environment State
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    """Full internal state of the environment (returned by state())."""

    task_name: str
    current_item_index: int
    total_items: int
    step_count: int
    max_steps: int
    done: bool
    cumulative_reward: float
    history: List[HistoryEntry] = Field(default_factory=list)
    last_action_error: Optional[str] = None
    score: Optional[float] = None
