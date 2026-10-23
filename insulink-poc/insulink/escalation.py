"""Escalation state machine. Pure logic with injected time, so it is fully testable.

In the real app this runs on the server (Cloud Functions timers), not on the phone.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Contact:
    name: str
    priority: int       # 1 = notified first


@dataclass(frozen=True)
class Notification:
    t: int
    recipient: str
    message: str


class Escalation:
    def __init__(self, contacts, ack_timeout_min=10, step_min=10):
        self.contacts = sorted(contacts, key=lambda c: c.priority)
        self.ack_timeout_min = ack_timeout_min
        self.step_min = step_min
        self.t0 = None
        self.kind = None
        self.acknowledged = False
        self.resolved = False
        self._notified = set()

    def trigger(self, t, kind):
        """Alert the user first. Contacts are only involved if the user does not respond."""
        self.t0, self.kind = t, kind
        self.acknowledged = self.resolved = False
        self._notified = set()
        return [Notification(t, "user", f"{kind}: please acknowledge")]

    def acknowledge(self, t):
        self.acknowledged = True

    def resolve(self, t):
        self.resolved = True

    def tick(self, t):
        """Call periodically. Returns notifications that became due at time t."""
        if self.t0 is None or self.acknowledged or self.resolved:
            return []
        elapsed = t - self.t0
        if elapsed < self.ack_timeout_min:
            return []
        due = 1 + (elapsed - self.ack_timeout_min) // self.step_min
        out = []
        for c in self.contacts[: int(due)]:
            if c.name not in self._notified:
                self._notified.add(c.name)
                out.append(Notification(t, c.name, f"{self.kind}: user has not responded"))
        return out

    def not_okay(self, t):
        """Manual 'I'm Not Okay': notify every contact immediately."""
        out = []
        for c in self.contacts:
            if c.name not in self._notified:
                self._notified.add(c.name)
                out.append(Notification(t, c.name, "User pressed 'I'm Not Okay'"))
        return out
