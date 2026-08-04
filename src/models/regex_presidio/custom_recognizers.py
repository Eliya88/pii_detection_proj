"""Custom Presidio recognizers for entity types not covered by the built-in set."""
from presidio_analyzer import Pattern, PatternRecognizer


class EmailRecognizer(PatternRecognizer):
    """Matches email addresses — replaces Presidio's built-in which misses many formats."""

    PATTERNS = [
        Pattern("EMAIL_FULL", r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}", 0.95),
    ]
    CONTEXT = ["email", "e-mail", "mail", "contact", "reach"]

    def __init__(self):
        super().__init__(
            supported_entity="EMAIL_ADDRESS",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
        )


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
    return [EmailRecognizer(), UsernameRecognizer()]
