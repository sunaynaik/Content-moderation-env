---
title: Content Moderation Env
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Content Moderation Environment

An **OpenEnv-compatible** environment that simulates a realistic content moderation workflow used by social platforms, forums, and community apps.  An AI agent reviews user-generated content — posts, comments, product reviews, messages — and takes moderation actions based on deterministic policy rules.

---

## Motivation

Content moderation is one of the most consequential real-world applications of AI decision-making.  Moderators must process high volumes of content, distinguish subtle policy violations from legitimate speech, and handle ambiguous borderline cases.  This environment provides a structured, deterministic testbed for evaluating how well an LLM-based agent can replicate human-level moderation judgment across three increasing levels of difficulty.

---

## Tasks

| # | Task | Difficulty | Action | Description |
|---|------|-----------|--------|-------------|
| 1 | `classification` | Easy | `classify` | Label content as **safe**, **spam**, or **unsafe** |
| 2 | `violation_detection` | Medium | `flag` | Identify the specific violation: **harassment**, **hate\_speech**, **scam**, or **none** |
| 3 | `moderation_decision` | Hard | `route` | Route content to **approve**, **reject**, or **escalate** (includes borderline cases) |

### Task 1 — Classification (Easy)

Surface-level categorization.  Most items can be classified from keywords alone.

### Task 2 — Violation Detection (Medium)

Requires reading comprehension to distinguish harassment from hate speech from scam content.

### Task 3 — Moderation Decision (Hard)

Requires nuanced judgment.  Borderline items (e.g. unverified health claims, emotional venting, polite self-promotion) must be escalated rather than approved or rejected outright.

---

## Project Structure

```
content-moderation-env/
├── env/
│   ├── __init__.py          # Package exports
│   ├── models.py            # Pydantic models (Observation, Action, Reward, State)
│   ├── data.py              # Deterministic dataset (20 items)
│   ├── tasks.py             # Task configs and policy contexts
│   ├── graders.py           # Deterministic scoring graders
│   └── environment.py       # Core env: reset(), step(), state()
├── inference.py             # LLM-driven agent (OpenAI client)
├── openenv.yaml             # OpenEnv manifest
├── Dockerfile               # Container for HF Spaces
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Observation Space

Each observation returned by `reset()` and `step()` includes:

| Field | Type | Description |
|-------|------|-------------|
| `task_name` | `str` | Current task identifier |
| `content_id` | `str` | Unique ID of the content item |
| `content_text` | `str` | The raw user-generated text |
| `policy_context` | `str` | Policy instructions for the current task |
| `available_actions` | `List[str]` | Valid action values |
| `step_count` | `int` | Steps taken so far |
| `remaining_items` | `int` | Items left to review |
| `history` | `List[HistoryEntry]` | Previous actions and rewards |
| `last_action_error` | `Optional[str]` | Error from the last invalid action |

Ground-truth labels are **never** revealed in observations.

---

## Action Space

Actions are typed Pydantic models with one branch per task:

### Classification
```json
{"action_type": "classify", "label": "safe|spam|unsafe"}
```

### Violation Detection
```json
{"action_type": "flag", "violation_type": "harassment|hate_speech|scam|none"}
```

### Moderation Decision
```json
{"action_type": "route", "decision": "approve|reject|escalate", "reason": "optional explanation"}
```

Invalid actions receive a **-0.1** penalty and the item is not advanced.

---

## Reward Design

Rewards are **incremental** (per-step) and **deterministic**.

### Classification
| Outcome | Reward |
|---------|--------|
| Correct label | +1.0 |
| Borderline partial match | +0.5 |
| Wrong label | -0.3 |
| Invalid action | -0.1 |

### Violation Detection
| Outcome | Reward |
|---------|--------|
| Correct violation type | +1.0 |
| Flagged violation but wrong type | +0.5 |
| Wrong / missed violation | -0.4 |
| Invalid action | -0.1 |

### Moderation Decision
| Outcome | Reward |
|---------|--------|
| Correct decision | +1.0 |
| Correct + good reasoning keywords | +1.5 |
| Approved harmful content | -0.5 |
| Wrong routing | -0.3 |
| Invalid action | -0.1 |

### Penalties
- Exceeding the 50-step maximum ends the episode
- Invalid actions cost -0.1 per occurrence

---

## Grader Logic

Each task has a dedicated grader in `graders.py` that produces a final score in **[0.0, 1.0]**:

- **ClassificationGrader**: Exact-match accuracy with partial credit for borderline pairs
- **ViolationGrader**: Exact-match with 0.25 credit for flagging the wrong violation type
- **ModerationGrader**: Weighted scoring that penalizes dangerous misses (approving harmful content) more heavily

All grading is deterministic and reproducible.

---

## Setup & Run Locally

### Prerequisites
- Python 3.10+
- A Hugging Face API token (or any OpenAI-compatible API key)

### Install dependencies
```bash
pip install -r requirements.txt
```

### Set environment variables
```bash
export HF_TOKEN="hf_your_token_here"
export API_BASE_URL="https://api-inference.huggingface.co/v1"   # optional
export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"          # optional
```

### Run a task
```bash
# Easy task
python inference.py classification

# Medium task
python inference.py violation_detection

# Hard task
python inference.py moderation_decision
```

---

## Run in Docker

### Build
```bash
docker build -t content-moderation-env .
```

### Run
```bash
docker run -e HF_TOKEN="hf_your_token" content-moderation-env python inference.py classification
docker run -e HF_TOKEN="hf_your_token" content-moderation-env python inference.py violation_detection
docker run -e HF_TOKEN="hf_your_token" content-moderation-env python inference.py moderation_decision
```

Target constraints: **2 vCPU / 8 GB RAM** — the container uses no heavy model downloads and runs purely via API calls.

---

## Baseline Performance

Expected scores with a typical instruction-tuned model (e.g. Mistral-7B-Instruct):

| Task | Expected Score | Notes |
|------|---------------|-------|
| `classification` | ~0.85–0.95 | Most items have clear surface cues |
| `violation_detection` | ~0.70–0.85 | Requires distinguishing violation types |
| `moderation_decision` | ~0.55–0.75 | Borderline items are deliberately ambiguous |

---

## Example Output

```
[START] task=classification env=content-moderation-env model=mistralai/Mistral-7B-Instruct-v0.3
[STEP]  step=1 action=classify:safe reward=1.00 done=false error=null
[STEP]  step=2 action=classify:safe reward=1.00 done=false error=null
[STEP]  step=3 action=classify:safe reward=1.00 done=false error=null
[STEP]  step=4 action=classify:safe reward=1.00 done=false error=null
[STEP]  step=5 action=classify:spam reward=1.00 done=false error=null
...
[STEP]  step=20 action=classify:safe reward=1.00 done=true error=null
[END]   success=true steps=20 rewards=1.00,1.00,1.00,...,1.00
```

---

## OpenEnv Compatibility

This environment follows the OpenEnv specification:

- **`reset()`** → returns typed `Observation`
- **`step(action)`** → returns `(Observation, Reward, done, info)`
- **`state()`** → returns `EnvironmentState`
- All models are **Pydantic v2** `BaseModel` subclasses
- `openenv.yaml` declares metadata, tasks, and environment variables
- Inference output follows the `[START]` / `[STEP]` / `[END]` format exactly
- All rewards and grading are **fully deterministic**

---

## License

MIT
