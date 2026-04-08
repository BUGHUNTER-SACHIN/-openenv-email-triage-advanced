from fastapi import FastAPI
from email_env import EmailEnv
from models import Action

app = FastAPI()

env = EmailEnv("medium")


@app.get("/")
def root():
    return {"status": "Email Env Running"}


@app.post("/reset")
async def reset():
    result = await env.reset()
    return result


@app.post("/step")
async def step(action: Action):
    result = await env.step(action)
    return result


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)



if __name__ == "__main__":
    main()