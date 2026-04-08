from fastapi import FastAPI
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from email_env import EmailEnv
from models import Action

app = FastAPI()

env = EmailEnv("medium")


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/reset")
async def reset():
    return await env.reset()


@app.post("/step")
async def step(action: Action):
    return await env.step(action)


@app.get("/state")
async def state():
    return env.state()

def main():
    return app