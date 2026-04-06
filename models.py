from pydantic import BaseModel
from typing import Optional


class Email(BaseModel):
    id: int
    subject: str
    body: str
    sender: str
    priority: str  # low / medium / high


class Observation(BaseModel):
    email: Email
    step: int


class Action(BaseModel):
    email_id: int
    action_type: str  # reply / archive / escalate / mark_spam
    reply_text: Optional[str] = None


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict = {}