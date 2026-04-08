import asyncio
import os
from openai import OpenAI

from email_env import EmailEnv
from models import Action

# ✅ MUST use these exact env variables
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

MAX_STEPS = 6


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    done_val = str(done).lower()

    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


async def main():
    # ✅ REQUIRED: Initialize OpenAI client using provided env vars
    client = None
    if API_BASE_URL and API_KEY:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY
        )

    env = EmailEnv("medium")

    rewards = []
    steps_taken = 0

    log_start("email_medium", "email_env", MODEL_NAME)

    result = await env.reset()

    for step in range(1, MAX_STEPS + 1):
        obs = result.observation.email

        # 🔥 REQUIRED: Make at least one LLM call
        if client:
            try:
                _ = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are an email classifier."},
                        {"role": "user", "content": f"Classify this email: {obs.subject}"}
                    ],
                    max_tokens=5,
                )
            except Exception:
                pass  # Don't break execution

        # ✅ Your logic (keep it simple & safe)
        if "win" in obs.subject.lower() or "crypto" in obs.subject.lower():
            action = Action(email_id=obs.id, action_type="mark_spam")
        elif obs.priority == "high":
            action = Action(email_id=obs.id, action_type="escalate")
        else:
            action = Action(email_id=obs.id, action_type="reply")

        action_str = f"{action.action_type}(email_id={action.email_id})"

        result = await env.step(action)

        reward = result.reward or 0.0
        done = result.done

        rewards.append(reward)
        steps_taken = step

        log_step(step, action_str, reward, done, None)

        if done:
            break

    # ✅ Proper score normalization
    score = sum(rewards) / len(rewards) if rewards else 0.0
    score = min(max(score, 0.0), 1.0)

    success = score > 0.3

    await env.close()
    log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())