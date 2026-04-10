"""inference.py — LLM-driven agent for the Content Moderation Environment.

Uses the OpenAI client to send observations to an LLM and parse moderation
actions.  Prints exactly the required stdout format (START / STEP / END).

Environment variables:
  API_BASE_URL  — base URL for the OpenAI-compatible API
  MODEL_NAME    — model identifier
  HF_TOKEN      — API key (required)
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from typing import List, Optional

from openai import OpenAI

from env.environment import ContentModerationEnv
from env.models import (
    Action,
    ActionType,
    ClassificationLabel,
    ModerationDecision,
    Observation,
    ViolationType,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("[END]   success=false steps=0 rewards=", flush=True)
    sys.exit(1)

BENCHMARK_NAME = "content-moderation-env"

# ---------------------------------------------------------------------------
# Action parsing helpers
# ---------------------------------------------------------------------------

VALID_LABELS = {l.value for l in ClassificationLabel}
VALID_VIOLATIONS = {v.value for v in ViolationType}
VALID_DECISIONS = {d.value for d in ModerationDecision}


def _build_prompt(obs: Observation) -> tuple[str, str]:
    """Build a system + user prompt from the observation."""
    system = (
        "You are an AI content moderation agent. You will be given a piece of "
        "user-generated content and a moderation policy. You must respond with "
        "a JSON object representing your moderation action.\n\n"
        "Rules:\n"
        "- Respond ONLY with a valid JSON object, no surrounding text.\n"
        "- Do NOT include markdown fences.\n"
    )

    task = obs.task_name
    if task == "classification":
        system += (
            '- For classification, respond with: {"action_type":"classify","label":"<safe|spam|unsafe>"}\n'
        )
    elif task == "violation_detection":
        system += (
            '- For violation detection, respond with: {"action_type":"flag","violation_type":"<harassment|hate_speech|scam|none>"}\n'
        )
    elif task == "moderation_decision":
        system += (
            '- For moderation decision, you may EITHER route the content OR gather more context.\n'
            '- To route, respond with: {"action_type":"route","decision":"<approve|reject|escalate>","reason":"<brief reason citing policy>"}\n'
            '- To review the hidden author history before routing, respond with: {"action_type":"investigate","investigate_target":"author_context"}\n'
        )

    user = (
        f"Policy:\n{obs.policy_context}\n\n"
        f"Content (id={obs.content_id}):\n\"{obs.content_text}\"\n\n"
        f"Available actions: {obs.available_actions}\n\n"
    )
    
    if getattr(obs, "investigation_result", None):
        user += f"--- NEW AUTHOR CONTEXT ---\n{obs.investigation_result}\n\n"
        
    user += "Respond with the JSON action only."

    return system, user


def _parse_action(raw: str, task: str) -> Optional[Action]:
    """Try to parse the LLM output into a valid Action."""
    # Strip markdown fences if present
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from the text
        match = re.search(r"\{[^}]+\}", raw)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    try:
        return Action(**data)
    except Exception:
        return None


def _fallback_action(task: str) -> Action:
    """Return a safe fallback action for the given task."""
    if task == "classification":
        return Action(action_type=ActionType.CLASSIFY, label=ClassificationLabel.SAFE)
    if task == "violation_detection":
        return Action(action_type=ActionType.FLAG, violation_type=ViolationType.NONE)
    if task == "moderation_decision":
        return Action(action_type=ActionType.ROUTE, decision=ModerationDecision.ESCALATE, reason="Uncertain, escalating for review.")
    raise ValueError(f"Unknown task: {task}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(task_name: str = "classification") -> None:
    """Run one full episode of the given task and print structured output."""
    rewards: List[float] = []
    steps = 0
    success = False

    print(f"[START] task={task_name} env={BENCHMARK_NAME} model={MODEL_NAME}", flush=True)

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        env = ContentModerationEnv(task_name=task_name)
        obs = env.reset()
        done = False
        score_val = 0.0

        while not done:
            system_prompt, user_prompt = _build_prompt(obs)

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=256,
                )
                raw_output = response.choices[0].message.content or ""
            except Exception:
                raw_output = ""

            action = _parse_action(raw_output, task_name)
            if action is None:
                action = _fallback_action(task_name)

            obs, reward, done, info = env.step(action)
            steps += 1
            rewards.append(reward.value)

            if done:
                score_val = info.get("score", 0.0)

            error_str = info.get("last_action_error") or "null"
            action_str = _action_to_str(action)
            done_str = "true" if done else "false"

            print(
                f"[STEP] step={steps} action={action_str} "
                f"reward={reward.value:.2f} done={done_str} error={error_str}",
                flush=True,
            )

        success = True

    except Exception:
        traceback.print_exc(file=sys.stderr)
        score_val = 0.0

    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else ""
    success_str = "true" if success else "false"
    print(f"[END]   success={success_str} steps={steps} rewards={rewards_str} score={score_val:.4f}", flush=True)


def _action_to_str(action: Action) -> str:
    """Compact string representation of an action for logging."""
    if action.action_type == ActionType.CLASSIFY:
        return f"classify:{action.label.value if action.label else 'None'}"
    if action.action_type == ActionType.FLAG:
        return f"flag:{action.violation_type.value if action.violation_type else 'None'}"
    if action.action_type == ActionType.ROUTE:
        decision = action.decision.value if action.decision else "None"
        return f"route:{decision}"
    if action.action_type == ActionType.INVESTIGATE:
        return f"investigate:{action.investigate_target}"
    return f"unknown:{action.action_type.value}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(sys.argv[1])
    else:
        # If no specific task provided, run all 3 sequentially so the validator
        # can correctly detect that all 3 tasks have valid grading pipelines!
        for t in ["classification", "violation_detection", "moderation_decision"]:
            run(t)
            print("-" * 50, flush=True)
