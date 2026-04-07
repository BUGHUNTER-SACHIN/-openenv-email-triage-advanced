from models import Observation, StepResult, Email
from tasks import TASKS


class EmailEnv:

    def __init__(self, task_name="easy"):
        self.task_name = task_name
        self.emails = TASKS[task_name]
        self.index = 0
        self.step_count = 0
        self.done = False

    async def reset(self):
        self.index = 0
        self.step_count = 0
        self.done = False

        email = Email(**self.emails[self.index])

        return StepResult(
            observation=Observation(email=email, step=0),
            reward=0.0,
            done=False,
        )

    async def step(self, action):
        current = self.emails[self.index]
        self.step_count += 1

        correct = current["label"]
        reward = 0.0

        # correct action
        if action.action_type == correct:
            reward += 0.6

        # partial credit
        if action.action_type in ["reply", "escalate"] and correct in ["reply", "escalate"]:
            reward += 0.2

        # priority bonus
        if current["priority"] == "high" and action.action_type == "escalate":
            reward += 0.2

        # penalty
        if action.action_type != correct:
            reward -= 0.3

        # time penalty
        reward -= 0.05 * self.step_count

        # clamp
        reward = min(max(reward, 0.0), 1.0)

        # next email
        self.index += 1
        if self.index >= len(self.emails):
            self.done = True
            next_email = current
        else:
            next_email = Email(**self.emails[self.index])

        return StepResult(
            observation=Observation(email=next_email , step=self.step_count),
            reward=reward,
            done=self.done,
        )

    def state(self):
        return {"index": self.index, "step": self.step_count}

    async def close(self):
        pass