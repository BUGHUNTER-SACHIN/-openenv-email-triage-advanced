from fastapi import FastAPI
from email_env import EmailEnv
from models import Action

app = FastAPI()
env = EmailEnv()


@app.get("/")
def home():
    return {"status": "Email Env Running"}

@app.post("/reset")
async def reset():
    result = await env.reset()
    return result.model_dump()


@app.post("/step")
async def step(action: Action):
    result = await env.step(action)
    return result.model_dump()

@app.get("/state")
async def state():
    return env.state()