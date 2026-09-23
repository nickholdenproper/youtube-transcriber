"""Shared fake model clients for unit tests."""


class FakeContextClient:
    """Records prompts and returns canned replies for both chunk and report calls."""

    def __init__(self):
        self.prompts = []

    def complete(self, messages, images=None):
        self.prompts.append((messages[0]["content"], images))
        return "# TL;DR\nTest record.\n\n# What happened\nNothing happened.\n"


class FakeTextClient:
    """Text client returning a canned reply; records prompts."""

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts = []

    def complete(self, messages, images=None):
        self.prompts.append(messages[0]["content"])
        return self.reply


class FakeBackend:
    """Vision backend stand-in used by frame / grid describing tests."""

    def __init__(self, batch_mode="ok"):
        self.calls = []
        self.batch_mode = batch_mode

    def complete(self, messages, images=None):
        self.calls.append((messages[0]["content"], list(images or [])))
        if self.batch_mode == "refuse-batches" and len(images or []) > 2:
            return "I cannot fulfill this request - I cannot provide a description."
        return "A busy dashboard with clear on-screen text."