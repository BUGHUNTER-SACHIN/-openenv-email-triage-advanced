from fastapi import FastAPI
from email_env import EmailEnv
from models import Action

app = FastAPI()
env = EmailEnv()


@app.post("/reset")
async def reset():
    result = await env.reset()
    return result.model_dump()


@app.post("/step")
async def step(action: Action):
    result = await env.step(action)
    return result.model_dump()