"""Deterministic Healthcare Misinformation Dataset.

Focuses on medical advice, alternative health scams, and crisis moderation.
Organized into easy, medium, and hard difficulty tiers with hidden author contexts
to support the INVESTIGATE tool-use trajectory.
"""

from env.models import (
    ContentItem,
    ClassificationLabel,
    ViolationType,
    ModerationDecision,
)

DATASET: list[ContentItem] = [
    # ---- EASY TIER ----
    
    ContentItem(
        id="med_001",
        text="What's a good home remedy for a mild headache? I usually just drink water and rest.",
        classification_label=ClassificationLabel.SAFE,
        violation_label=ViolationType.NONE,
        moderation_decision=ModerationDecision.APPROVE,
        policy_reason="Harmless question about general well-being.",
        difficulty="easy",
        author_context="Account created 5 years ago, 0 violations.",
        required_reasoning_concepts=["general wellbeing", "harmless"]
    ),
    ContentItem(
        id="med_002",
        text="LOSE 50 LBS IN 3 DAYS WITH THIS MIRACLE PILL. CLICK HERE: http://fake-pill-scam.com",
        classification_label=ClassificationLabel.SPAM,
        violation_label=ViolationType.SCAM,
        moderation_decision=ModerationDecision.REJECT,
        policy_reason="Blatant health product scam with unrealistic claims.",
        difficulty="easy",
        author_context="Account created 2 hours ago. Has posted 50 similar links.",
        required_reasoning_concepts=["unrealistic claims", "scam"]
    ),
    ContentItem(
        id="med_003",
        text="I always feel terrible on Mondays. Getting out of bed is a struggle.",
        classification_label=ClassificationLabel.SAFE,
        violation_label=ViolationType.NONE,
        moderation_decision=ModerationDecision.APPROVE,
        policy_reason="General expression of fatigue, not a medical crisis.",
        difficulty="easy",
        author_context="Verified user. Frequent poster in lifestyle communities.",
        required_reasoning_concepts=["general fatigue", "no medical claim"]
    ),
    ContentItem(
        id="med_004",
        text="Injecting household bleach is the ONLY way to protect yourself from the new virus!",
        classification_label=ClassificationLabel.UNSAFE,
        violation_label=ViolationType.HARASSMENT, # Or Hate speech/scam. We'll use HATE_SPEECH/SCAM/HARASSMENT. Wait, ViolationType doesn't have a MISINFO type! Let's map it to SCAM representing dangerous fake info.
        moderation_decision=ModerationDecision.REJECT,
        policy_reason="Promotes life-threatening internal ingestion of toxic chemicals.",
        difficulty="easy",
        author_context="Account flagged previously for self-harm promotion.",
        required_reasoning_concepts=["toxic", "life-threatening", "harm"]
    ),

    # ---- MEDIUM TIER ----
    
    ContentItem(
        id="med_005",
        text="I've been taking high doses of Vitamin C and honestly, I haven't been sick once all year. You all should try taking 5000mg a day instead of going to the doctor.",
        classification_label=ClassificationLabel.UNSAFE,
        violation_label=ViolationType.SCAM,
        moderation_decision=ModerationDecision.REJECT,
        policy_reason="Crosses from personal anecdote into giving explicit medical advice to bypass professional care.",
        difficulty="medium",
        author_context="Unverified user. No medical credentials listed.",
        required_reasoning_concepts=["medical advice", "bypassing professional care"]
    ),
    ContentItem(
        id="med_006",
        text="My doctor prescribed me this new antidepressant but the side effects are horrible. Anyone else experience extreme nausea with it?",
        classification_label=ClassificationLabel.SAFE,
        violation_label=ViolationType.NONE,
        moderation_decision=ModerationDecision.APPROVE,
        policy_reason="Discussing personal experience with diagnosed medication side effects is generally permitted.",
        difficulty="medium",
        author_context="Account age: 2 years. Active in mental health support groups.",
        required_reasoning_concepts=["personal experience", "side effects"]
    ),
    ContentItem(
        id="med_007",
        text="Cancer is just a fungus. I sell a specialized baking soda treatment that cures it entirely in 4 weeks. PM for pricing.",
        classification_label=ClassificationLabel.SPAM,
        violation_label=ViolationType.SCAM,
        moderation_decision=ModerationDecision.REJECT,
        policy_reason="Monetizing a fake cure for a terminal illness.",
        difficulty="medium",
        author_context="Has been banned on 3 alt accounts for medical fraud.",
        required_reasoning_concepts=["monetizing", "fake cure", "fraud"]
    ),
    ContentItem(
        id="med_008",
        text="I am feeling incredibly overwhelmed today and I just can't see the point of continuing anymore. I don't want to be here.",
        classification_label=ClassificationLabel.UNSAFE,
        violation_label=ViolationType.NONE,
        moderation_decision=ModerationDecision.ESCALATE,
        policy_reason="Borderline self-harm/crisis situation requiring human review and crisis response resources.",
        difficulty="medium",
        author_context="User has mentioned recent job loss. 0 previous safety flags.",
        required_reasoning_concepts=["crisis", "self-harm", "human review"]
    ),

    # ---- HARD TIER (Nuanced, requires investigate or deep reasoning) ----

    ContentItem(
        id="med_009",
        text="As a researcher at Johns Hopkins, my latest preprint shows that standard mRNA vaccines interact negatively with neuro-receptors. Stop taking them immediately until we peer-review.",
        classification_label=ClassificationLabel.UNSAFE,
        violation_label=ViolationType.SCAM,
        moderation_decision=ModerationDecision.ESCALATE,
        policy_reason="Appeals to authority while spreading widespread anti-vaccine panic; needs credential verification and human review.",
        difficulty="hard",
        author_context="IMPERSONATOR: Account originates from a known bot farm IP. No actual ties to Johns Hopkins.",
        required_reasoning_concepts=["impersonation", "verification", "panic"]
    ),
    ContentItem(
        id="med_010",
        text="I read an interesting article about how ancient civilizations used colloidal silver to purify water. Wonder if it would work for modern tap water?",
        classification_label=ClassificationLabel.SAFE,
        violation_label=ViolationType.NONE,
        moderation_decision=ModerationDecision.APPROVE,
        policy_reason="Harmless historical question, does not recommend ingesting or replacing modern medicine.",
        difficulty="hard",
        author_context="Historical trivia enthusiast. Completely clean record.",
        required_reasoning_concepts=["historical", "no ingestion recommendation"]
    ),
    ContentItem(
        id="med_011",
        text="I literally want to kill my pharmacist. They refused to refill my prescription because it was 'too early'. I'm going to wait outside their car tonight.",
        classification_label=ClassificationLabel.UNSAFE,
        violation_label=ViolationType.HARASSMENT,
        moderation_decision=ModerationDecision.REJECT,
        policy_reason="Actionable, specific threat of violence against a healthcare professional.",
        difficulty="hard",
        author_context="User has posted the literal address of the pharmacy in a previous deleted post.",
        required_reasoning_concepts=["specific threat", "violence", "actionable"]
    ),
    ContentItem(
        id="med_012",
        text="We are organizing a protest outside the local clinic tomorrow to demand better access to off-label treatments. Bring signs!",
        classification_label=ClassificationLabel.SAFE, # Though controversial, organizing medical protests is typically protected speech.
        violation_label=ViolationType.NONE,
        moderation_decision=ModerationDecision.ESCALATE,
        policy_reason="Political/activist organization. Borderline policy depending on platform's local clinic protection rules.",
        difficulty="hard",
        author_context="Verified grassroots community organizer. No history of violence.",
        required_reasoning_concepts=["protest", "protected speech", "activism"]
    )
]

# Helper to load by difficulty
def get_items_by_difficulty(diff: str) -> list[ContentItem]:
    return [item for item in DATASET if item.difficulty == diff]
