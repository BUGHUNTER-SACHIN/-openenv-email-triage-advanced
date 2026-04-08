<<<<<<< HEAD
# OpenEnv Email Triage Advanced

This project implements a real-world Email Triage environment using the OpenEnv framework. It simulates how support teams process incoming emails by classifying, prioritizing, and taking appropriate actions.

## 🚀 Features

- Multi-action workflow: reply, archive, escalate, mark_spam
- Priority-aware decision making (low, medium, high)
- Spam detection signals based on email content
- Step-based environment with realistic time penalties
- Dense reward system with partial credit for near-correct actions
- Deterministic grading with scores normalized between 0 and 1
- Three task levels: easy, medium, hard

## 🧠 Environment Design

### Observation
The agent receives one email per step, including:
- subject
- body
- sender
- priority level

### Actions
The agent must choose:
- reply
- archive
- escalate
- mark_spam

Optional reply text can be included.

### Reward Function
- Correct action: positive reward
- Partial intent match: partial reward
- High-priority escalation bonus
- Penalty for incorrect actions
- Time penalty to discourage inefficient behavior

## 📊 Tasks

- **Easy**: Clear spam vs normal emails
- **Medium**: Mixed intent with priority considerations
- **Hard**: Ambiguous and high-stakes scenarios

## ⚙️ Tech Stack

- Python
- FastAPI (for API endpoints)
- Pydantic (typed models)
- OpenAI Client (for inference)
- Docker (containerized deployment)

## 🔌 API Endpoints

- `POST /reset` → Initialize environment
- `POST /step` → Perform action

## ▶️ Running Locally

```bash
docker build -t email-env .
docker run -p 7860:7860 email-env
=======
---
title: Openenv Email Triage Advanced
emoji: 🐠
colorFrom: pink
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: 'An OpenEnv-compliant environment that simulates real-world '
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> c73315c65f7684f43bad4ff6a83d5afe6d7ecbad
