import asyncio
import os
from openai import OpenAI
from email_env import EmailEnv
from models import Action

# ✅ FIXED: Use API_KEY (not HF_TOKEN) as injected by the platform
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
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

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0,
    )

    action_text = response.choices[0].message.content.strip().lower()

    # Validate and fallback
    valid_actions = {"reply", "archive", "escalate", "mark_spam"}
    for valid in valid_actions:
        if valid in action_text:
            return valid

    return "reply"  # safe default


async def main():
    # ✅ FIXED: Always initialize the client using injected env vars
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = EmailEnv("medium")
    rewards = []
    steps_taken = 0

    log_start("email_medium", "email_env", MODEL_NAME)

    result = await env.reset()

    for step in range(1, MAX_STEPS + 1):
        obs = result.observation.email

        # ✅ FIXED: Use LLM through the proxy for every decision
        action_type = get_action_from_llm(client, obs)
        action = Action(email_id=obs.id, action_type=action_type)

        action_str = f"{action.action_type}(email_id={action.email_id})"

        result = await env.step(action)
        reward = result.reward or 0.0
        done = result.done

        rewards.append(reward)
        steps_taken = step

        log_step(step, action_str, reward, done, None)

        if done:
            break

    if len(rewards) > 0:
        score = sum(rewards) / len(rewards)
    else:
        score = 0.0

    score = min(max(score, 0.0), 1.0)
    success = score > 0.3

    await env.close()
    log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())