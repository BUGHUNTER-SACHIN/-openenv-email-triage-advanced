---
title: OpenEnv Email Triage Advanced
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
short_description: Advanced OpenEnv environment simulating real-world email triage with priority handling, spam detection, and escalation workflows.
---

# OpenEnv Email Triage Advanced

This project implements a real-world Email Triage environment using the OpenEnv framework. It simulates how support teams process incoming emails by classifying, prioritizing, and taking appropriate actions.

---

## 🚀 Features

- Multi-action workflow: reply, archive, escalate, mark_spam  
- Priority-aware decision making (low, medium, high)  
- Spam detection based on content patterns  
- Step-based environment with time penalties  
- Dense reward system with partial credit  
- Deterministic grading (scores between 0 and 1)  
- Three difficulty levels: easy → medium → hard  

---

## 🧠 Environment Design

### Observation
Each step provides an email with:
- subject  
- body  
- sender  
- priority  

---

### Actions
The agent must choose one:

- `reply`  
- `archive`  
- `escalate`  
- `mark_spam`  

Optional reply text can be included.

---

### Reward Function

- Correct action → positive reward  
- Partial match → partial reward  
- High-priority escalation bonus  
- Wrong action → penalty  
- Time penalty → discourages inefficiency  

👉 Designed for **deterministic reward shaping with partial credit**

---

## 📊 Tasks

- **Easy** → Clear spam vs normal emails  
- **Medium** → Mixed signals with priority  
- **Hard** → Ambiguous + high-risk cases  

---

## ⚙️ Tech Stack

- Python  
- FastAPI  
- Pydantic  
- OpenAI Client  
- Docker  

---

## 🔌 API Endpoints

- `POST /reset` → Initialize environment  
- `POST /step` → Perform action  

---

## ▶️ Run Locally

```bash
docker build -t email-env .
docker run -p 7860:7860 email-env