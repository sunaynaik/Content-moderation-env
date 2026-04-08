"""Deterministic dataset for the Content Moderation Environment.

Contains 20 content items with ground-truth labels for all three tasks.
The dataset covers safe, spam, unsafe, scam, harassment, hate-speech,
and borderline content to test classification, violation detection,
and moderation decision logic.
"""

from env.models import (
    ContentItem,
    ClassificationLabel,
    ViolationType,
    ModerationDecision,
)

DATASET: list[ContentItem] = [
    # ---- Safe content ----
    ContentItem(
        id="c001",
        text="Great tutorial, thanks for sharing this with the community!",
        classification_label=ClassificationLabel.SAFE,
        violation_label=ViolationType.NONE,
        moderation_decision=ModerationDecision.APPROVE,
        policy_reason="Positive, constructive comment with no policy issues.",
    ),
    ContentItem(
        id="c002",
        text="I really enjoyed today's live stream. The Q&A at the end was fantastic.",
        classification_label=ClassificationLabel.SAFE,
        violation_label=ViolationType.NONE,
        moderation_decision=ModerationDecision.APPROVE,
        policy_reason="Genuine appreciation with no harmful content.",
    ),
    ContentItem(
        id="c003",
        text="Could someone help me set up my development environment? I keep getting errors.",
        classification_label=ClassificationLabel.SAFE,
        violation_label=ViolationType.NONE,
        moderation_decision=ModerationDecision.APPROVE,
        policy_reason="Legitimate support request.",
    ),
    ContentItem(
        id="c004",
        text="Happy birthday! Hope you have an amazing day.",
        classification_label=ClassificationLabel.SAFE,
        violation_label=ViolationType.NONE,
        moderation_decision=ModerationDecision.APPROVE,
        policy_reason="Friendly, harmless message.",
    ),

    # ---- Spam content ----
    ContentItem(
        id="c005",
        text="Buy followers instantly! Cheapest rates guaranteed. DM for details.",
        classification_label=ClassificationLabel.SPAM,
        violation_label=ViolationType.SCAM,
        moderation_decision=ModerationDecision.REJECT,
        policy_reason="Promotional spam selling fake engagement metrics.",
    ),
    ContentItem(
        id="c006",
        text="FREE iPhone giveaway!!! Click the link in my bio NOW before it expires!!!",
        classification_label=ClassificationLabel.SPAM,
        violation_label=ViolationType.SCAM,
        moderation_decision=ModerationDecision.REJECT,
        policy_reason="Classic clickbait giveaway scam.",
    ),
    ContentItem(
        id="c007",
        text="Check out my store for amazing deals on handbags, shoes, and watches. 90% off everything!",
        classification_label=ClassificationLabel.SPAM,
        violation_label=ViolationType.SCAM,
        moderation_decision=ModerationDecision.REJECT,
        policy_reason="Unsolicited commercial promotion with unrealistic discounts.",
    ),

    # ---- Unsafe / harassment ----
    ContentItem(
        id="c008",
        text="You are completely worthless and nobody would care if you disappeared.",
        classification_label=ClassificationLabel.UNSAFE,
        violation_label=ViolationType.HARASSMENT,
        moderation_decision=ModerationDecision.REJECT,
        policy_reason="Targeted personal attack implying the person has no value.",
    ),
    ContentItem(
        id="c009",
        text="Shut up, you absolute moron. Nobody asked for your garbage opinion.",
        classification_label=ClassificationLabel.UNSAFE,
        violation_label=ViolationType.HARASSMENT,
        moderation_decision=ModerationDecision.REJECT,
        policy_reason="Direct insult and personal degradation.",
    ),

    # ---- Unsafe / hate speech ----
    ContentItem(
        id="c010",
        text="People from that country are all criminals and should be deported immediately.",
        classification_label=ClassificationLabel.UNSAFE,
        violation_label=ViolationType.HATE_SPEECH,
        moderation_decision=ModerationDecision.REJECT,
        policy_reason="Discriminatory generalization targeting a national group.",
    ),
    ContentItem(
        id="c011",
        text="That entire religion is a disease and its followers should be removed from society.",
        classification_label=ClassificationLabel.UNSAFE,
        violation_label=ViolationType.HATE_SPEECH,
        moderation_decision=ModerationDecision.REJECT,
        policy_reason="Dehumanizing language targeting a religious group.",
    ),

    # ---- Scam / phishing ----
    ContentItem(
        id="c012",
        text="URGENT: Your account will be suspended. Send your password to verify@securecheck.com to keep access.",
        classification_label=ClassificationLabel.UNSAFE,
        violation_label=ViolationType.SCAM,
        moderation_decision=ModerationDecision.REJECT,
        policy_reason="Phishing attempt trying to steal credentials.",
    ),
    ContentItem(
        id="c013",
        text="Congratulations! You have been selected to receive $5,000. Click here and enter your bank details.",
        classification_label=ClassificationLabel.SPAM,
        violation_label=ViolationType.SCAM,
        moderation_decision=ModerationDecision.REJECT,
        policy_reason="Advance-fee fraud / financial phishing.",
    ),

    # ---- Borderline / escalation ----
    ContentItem(
        id="c014",
        text="This herbal supplement cured my chronic illness in just three days. Big pharma doesn't want you to know.",
        classification_label=ClassificationLabel.SPAM,
        violation_label=ViolationType.SCAM,
        moderation_decision=ModerationDecision.ESCALATE,
        policy_reason="Unverified health claims that could mislead users but may be genuine personal experience.",
    ),
    ContentItem(
        id="c015",
        text="I think certain communities are over-represented in media and we should have an honest discussion about it.",
        classification_label=ClassificationLabel.SAFE,
        violation_label=ViolationType.NONE,
        moderation_decision=ModerationDecision.ESCALATE,
        policy_reason="Borderline opinion that touches sensitive identity topics; needs human review.",
    ),
    ContentItem(
        id="c016",
        text="Some people just don't belong in this profession. It's obvious when you look at who keeps failing.",
        classification_label=ClassificationLabel.UNSAFE,
        violation_label=ViolationType.HARASSMENT,
        moderation_decision=ModerationDecision.ESCALATE,
        policy_reason="Veiled discrimination; not an explicit slur but implies group exclusion.",
    ),
    ContentItem(
        id="c017",
        text="I'm so angry I could scream. This company ruined my life and I want everyone to know.",
        classification_label=ClassificationLabel.SAFE,
        violation_label=ViolationType.NONE,
        moderation_decision=ModerationDecision.ESCALATE,
        policy_reason="Emotional venting that could be interpreted as a threat; needs human review.",
    ),

    # ---- More safe ----
    ContentItem(
        id="c018",
        text="Does anyone know a good recipe for sourdough bread? I've been trying for weeks.",
        classification_label=ClassificationLabel.SAFE,
        violation_label=ViolationType.NONE,
        moderation_decision=ModerationDecision.APPROVE,
        policy_reason="Harmless off-topic question.",
    ),
    ContentItem(
        id="c019",
        text="Just finished reading the documentation. The new API looks really clean.",
        classification_label=ClassificationLabel.SAFE,
        violation_label=ViolationType.NONE,
        moderation_decision=ModerationDecision.APPROVE,
        policy_reason="Constructive technical feedback.",
    ),

    # ---- Borderline spam ----
    ContentItem(
        id="c020",
        text="I made a video about this topic on my channel. Not trying to self-promote, just thought it might help.",
        classification_label=ClassificationLabel.SAFE,
        violation_label=ViolationType.NONE,
        moderation_decision=ModerationDecision.ESCALATE,
        policy_reason="Self-promotion but framed politely; borderline between helpful sharing and spam.",
    ),
]
