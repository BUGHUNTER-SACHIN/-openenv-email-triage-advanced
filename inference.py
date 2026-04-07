import asyncio
import os
from openai import OpenAI

from email_env import EmailEnv
from models import Action

API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

MAX_STEPS = 6


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


async def main():
    client = None
    if API_KEY:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EmailEnv("medium")

    rewards = []
    log_start("email_medium", "email_env", MODEL_NAME)

    result = await env.reset()

    for step in range(1, MAX_STEPS + 1):
        obs = result.observation.email

        # simple safe heuristic (NOT relying on LLM parsing)
        if "win" in obs.subject.lower() or "crypto" in obs.subject.lower():
            action = Action(email_id=obs.id, action_type="mark_spam")
        elif obs.priority == "high":
            action = Action(email_id=obs.id, action_type="escalate")
        else:
            action = Action(email_id=obs.id, action_type="reply")

        result = await env.step(action)

        reward = result.reward
        done = result.done

        rewards.append(reward)

        log_step(step, str(action), reward, done, None)

        if done:
            break

    score = min(max(sum(rewards) / MAX_STEPS, 0.0), 1.0)
    success = score > 0.3

    await env.close()
    log_end(success, step, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())