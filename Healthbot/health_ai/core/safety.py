"""
safety.py — Red flag detection and disclaimer injection for Health AI v3.

Red flags are medical emergencies or concerning query patterns that should
trigger an urgent notice appended to Dr. Aria's response.

This module is the SINGLE SOURCE OF TRUTH for DISCLAIMER and URGENT_NOTICE.
Other modules (character.py, server.py) import from here.
"""

import re

_URGENT_KEYWORDS = frozenset([
    # Cardiac
    "heart attack", "cardiac arrest", "chest pain", "chest tightness",
    "palpitations", "irregular heartbeat",
    # Neurological
    "stroke", "seizure", "unconscious", "unresponsive", "fainting",
    "sudden confusion", "can't speak", "cannot speak",
    # Respiratory
    "can't breathe", "cannot breathe", "shortness of breath", "choking",
    "stopped breathing",
    # Bleeding / trauma
    "severe bleeding", "heavy bleeding", "coughing blood", "vomiting blood",
    "blood in urine",
    # Mental health emergencies
    "suicide", "suicidal", "self harm", "self-harm", "overdose", "poisoning",
    "want to die", "kill myself",
    # General emergency
    "emergency", "ambulance", "call 911", "call 999", "call 112",
])

# Compiled word-boundary regex for accurate matching (no substring false-positives)
_URGENT_REGEX = re.compile(
    r'\b(?:' + '|'.join(re.escape(k) for k in sorted(_URGENT_KEYWORDS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)

# Disclaimer appended to EVERY response
DISCLAIMER = (
    "\n\n---\n"
    "*Dr. Aria is an AI assistant, not a licensed doctor. "
    "This information is for educational purposes only. "
    "Always consult a qualified healthcare professional for medical advice, "
    "diagnosis, or treatment.*"
)

# Appended ONLY when urgent keywords are detected
URGENT_NOTICE = (
    "\n\n**URGENT — Please seek immediate medical attention.** "
    "Some of the symptoms or values you mentioned may indicate a medical emergency. "
    "Call emergency services (911 / 999 / 112) or go to the nearest hospital now. "
    "Do not wait."
)


def detect_red_flags(query: str) -> bool:
    """
    Return True if the query contains any urgent/emergency keyword.

    Uses word-boundary regex to avoid false positives from substring matches.

    Args:
        query: The raw user query string.

    Returns:
        True if an urgent keyword is found, False otherwise.
    """
    return bool(_URGENT_REGEX.search(query))


# Response returned when a prompt injection/bypass attempt is detected
BYPASS_ATTEMPT_RESPONSE = (
    "I'm **Dr. Aria**, your health assistant. "
    "I cannot ignore or bypass my instructions to act as a health chatbot. "
    "Please ask me a health-related question."
)


# Compiled word-boundary regex for detecting instruction bypass / prompt injection attempts
_INJECTION_REGEX = re.compile(
    r'\b(?:'
    r'ignore\s+(?:all\s+|previous\s+|system\s+)?instructions|'
    r'bypass\s+(?:all\s+|previous\s+|system\s+)?instructions|'
    r'system\s+override|'
    r'forget\s+(?:all\s+|previous\s+|your\s+)?instructions|'
    r'do\s+not\s+act\s+as|'
    r'stop\s+(?:being|acting\s+as)\s+(?:a\s+)?(?:chatbot|bot|health\s+assistant|medical\s+assistant|doctor|aria)|'
    r'you\s+are\s+now\s+a'
    r')\b',
    re.IGNORECASE
)


def detect_prompt_injection(query: str) -> bool:
    """
    Return True if the query contains any pattern attempting to bypass or ignore instructions.
    """
    return bool(_INJECTION_REGEX.search(query))


def apply_safety_layer(response: str, query: str) -> str:
    """
    Append the disclaimer and (if needed) urgent notice to a response, and strip emojis.

    Args:
        response: The raw LLM-generated response string.
        query:    The original user query (used for red flag detection).

    Returns:
        The response with safety text appended and emojis stripped.
    """
    if detect_red_flags(query):
        response += URGENT_NOTICE
    response += DISCLAIMER

    # Globally strip any generated emojis from the final output text
    emoji_regex = re.compile(r'[\U00010000-\U0010ffff\u2600-\u27bf\u2300-\u23ff\u2b50]')
    response = emoji_regex.sub("", response).replace("\ufe0f", "")

    return response
