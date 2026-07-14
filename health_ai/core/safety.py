"""
safety.py — Red flag detection and disclaimer injection for Health AI v3.

Red flags are medical emergencies or concerning query patterns that should
trigger an urgent notice appended to Dr. Aria's response.
"""

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

# Disclaimer appended to EVERY response
DISCLAIMER = (
    "\n\n---\n"
    "⚕️ *Dr. Aria is an AI assistant, not a licensed doctor. "
    "This information is for educational purposes only. "
    "Always consult a qualified healthcare professional for medical advice, "
    "diagnosis, or treatment.*"
)

# Appended ONLY when urgent keywords are detected
URGENT_NOTICE = (
    "\n\n🚨 **URGENT — Please seek immediate medical attention.** "
    "Some of the symptoms or values you mentioned may indicate a medical emergency. "
    "Call emergency services or go to the nearest hospital now. "
    "Do not wait."
)


def detect_red_flags(query: str) -> bool:
    """
    Return True if the query contains any urgent/emergency keyword.

    Args:
        query: The raw user query string.

    Returns:
        True if an urgent keyword is found, False otherwise.
    """
    q = query.lower()
    return any(k in q for k in _URGENT_KEYWORDS)


def apply_safety_layer(response: str, query: str) -> str:
    """
    Append the disclaimer and (if needed) urgent notice to a response.

    Args:
        response: The raw LLM-generated response string.
        query:    The original user query (used for red flag detection).

    Returns:
        The response with safety text appended.
    """
    if detect_red_flags(query):
        response += URGENT_NOTICE
    response += DISCLAIMER
    return response
