"""Deterministic graders for each task in the Content Moderation Environment.

Each grader evaluates a list of (predicted, ground_truth) pairs and returns
a score in [0.0, 1.0].  The grading is fully deterministic and reproducible.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Base grader
# ---------------------------------------------------------------------------

class BaseGrader:
    """Abstract base grader."""

    def grade(self, results: List[Tuple[str, str]]) -> float:
        """Return a score in [0.0, 1.0] given (predicted, expected) pairs."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Classification grader  (Task 1 — easy)
# ---------------------------------------------------------------------------

class ClassificationGrader(BaseGrader):
    """Grades classification accuracy with partial credit for borderline items.

    Scoring per item:
      - exact match       → 1.0
      - borderline match  → 0.5  (e.g. predicting "spam" when truth is "unsafe"
        on certain ambiguous items)
      - wrong             → 0.0

    Final score = mean of per-item scores.
    """

    # Pairs where partial credit is reasonable (order-independent)
    BORDERLINE_PAIRS = frozenset({
        frozenset({"spam", "unsafe"}),
        frozenset({"safe", "spam"}),
    })

    def grade(self, results: List[Tuple[str, str]]) -> float:
        if not results:
            return 0.0
        total = 0.0
        for predicted, expected in results:
            if predicted == expected:
                total += 1.0
            elif frozenset({predicted, expected}) in self.BORDERLINE_PAIRS:
                total += 0.5
            # else: 0.0
        return round(total / len(results), 4)


# ---------------------------------------------------------------------------
# Violation detection grader  (Task 2 — medium)
# ---------------------------------------------------------------------------

class ViolationGrader(BaseGrader):
    """Grades violation detection accuracy.

    Scoring per item:
      - exact match                     → 1.0
      - predicted "none" when truth is a violation → 0.0
      - predicted a violation when truth is "none" → 0.0
      - wrong violation type but at least flagged  → 0.25
      - wrong                           → 0.0

    Final score = mean of per-item scores.
    """

    def grade(self, results: List[Tuple[str, str]]) -> float:
        if not results:
            return 0.0
        total = 0.0
        for predicted, expected in results:
            if predicted == expected:
                total += 1.0
            elif expected != "none" and predicted != "none":
                # Wrong specific violation but at least flagged something
                total += 0.25
            # else: 0.0
        return round(total / len(results), 4)


# ---------------------------------------------------------------------------
# Moderation decision grader  (Task 3 — hard)
# ---------------------------------------------------------------------------

class ModerationGrader(BaseGrader):
    """Grades moderation decisions with reasoning bonus.

    Scoring per item:
      - exact match                            → 1.0
      - predicted "escalate" when truth is "reject"   → 0.4  (cautious)
      - predicted "reject" when truth is "escalate"    → 0.3  (over-cautious)
      - predicted "approve" when truth is not "approve" → 0.0  (dangerous miss)
      - other mismatch                         → 0.1

    Final score = mean of per-item scores.
    """

    def grade(
        self,
        results: List[Tuple[str, str]],
        reasons: Optional[List[Optional[str]]] = None,
    ) -> float:
        if not results:
            return 0.0
        total = 0.0
        for i, (predicted, expected) in enumerate(results):
            if predicted == expected:
                item_score = 1.0
            elif predicted == "escalate" and expected == "reject":
                item_score = 0.4
            elif predicted == "reject" and expected == "escalate":
                item_score = 0.3
            elif predicted == "approve" and expected != "approve":
                item_score = 0.0  # dangerous miss
            else:
                item_score = 0.1
            total += item_score
        return round(total / len(results), 4)
