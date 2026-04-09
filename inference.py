import asyncio
import os
from typing import List, Optional
from openai import OpenAI
from email_env import EmailEnv
from models import Action

# ✅ Match sample: try HF_TOKEN first, then API_KEY
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

MAX_STEPS = 6
TASK_NAME = "email_medium"
BENCHMARK = "email_env"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def get_action_from_llm(client: OpenAI, obs) -> str:
    """Call the LLM through the proxy to decide the action."""
    prompt = f"""You are an email triage assistant. Given this email, choose exactly one action.

Email:
- Subject: {obs.subject}
- Body: {obs.body}
- Sender: {obs.sender}
- Priority: {obs.priority}

Choose ONLY one of these actions (output just the action word, nothing else):
- reply       (normal emails needing a response)
- archive     (low-priority, informational emails)
- escalate    (high-priority or urgent issues)
- mark_spam   (spam or promotional emails)

Action:"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
            stream=False,
        )
        action_text = (completion.choices[0].message.content or "").strip().lower()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        action_text = "reply"

    # Validate — order matters (check mark_spam before spam)
    for valid in ["mark_spam", "escalate", "archive", "reply"]:
        if valid in action_text:
            return valid

    return "reply"  # safe default


async def main() -> None:
    # ✅ Client init matching sample script pattern
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = EmailEnv("medium")
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs = result.observation.email

            action_type = get_action_from_llm(client, obs)
            action = Action(email_id=obs.id, action_type=action_type)
            action_str = f"{action.action_type}(email_id={action.email_id})"

            result = await env.step(action)
            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score > 0.3

    finally:
        # ✅ Match sample: always close env and always emit [END]
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())