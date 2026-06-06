"""Custom Presidio recognizers for entity types not covered by the built-in set."""
from presidio_analyzer import Pattern, PatternRecognizer


class UsernameRecognizer(PatternRecognizer):
    """Matches @handle social-media usernames only."""

    PATTERNS = [
        Pattern("USERNAME_HANDLE", r"@[a-zA-Z0-9_]{2,30}\b", 0.85),
    ]
    CONTEXT = ["username", "user name", "login", "account", "handle", "screen name", "user"]

    def __init__(self):
        super().__init__(
            supported_entity="USERNAME",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
        )


def get_custom_recognizers() -> list:
    return [UsernameRecognizer()]
