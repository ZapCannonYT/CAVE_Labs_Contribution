"""
test_classify_intent.py — Unit tests for classify_intent and keyword matching.

Covers:
  - Correct intent routing for urgent, mental_health, symptom, lab, prescription, general
  - Priority ordering (urgent > mental_health > symptom)
  - False-positive prevention (word-boundary regex blocks substring matches)
  - Greeting / farewell / off_topic classification
"""

import pytest
from health_ai.core.character import (
    classify_intent,
    detect_urgent,
    is_health_related,
)


# ── Bug 1: urgent intent is no longer dead code ─────────────────────────────

class TestUrgentIntent:
    """Verify urgent queries are classified correctly (checked FIRST)."""

    @pytest.mark.parametrize("query", [
        "I want to kill myself",
        "I'm having a heart attack",
        "call 911 now",
        "he is choking and can't breathe",
        "my friend is unconscious",
        "she took an overdose of pills",
        "I think I'm having a stroke",
        "there is severe bleeding from the wound",
        "I want to die",
        "end my life please",
    ])
    def test_urgent_queries_classified_as_urgent(self, query):
        assert classify_intent(query) == "urgent"

    def test_urgent_beats_symptom(self):
        """Crisis phrases like 'chest pain' should be urgent, not symptom."""
        assert classify_intent("I have severe chest pain and can't breathe") == "urgent"

    def test_suicidal_not_routed_to_symptom(self):
        """'kill myself' must NEVER route to symptom (home-remedy advice)."""
        assert classify_intent("I want to kill myself") == "urgent"
        assert classify_intent("I feel suicidal") == "urgent"

    def test_detect_urgent_function(self):
        assert detect_urgent("I think I'm having a heart attack") is True
        assert detect_urgent("I had a good breakfast") is False


# ── Bug 1: mental_health intent is no longer dead code ───────────────────────

class TestMentalHealthIntent:
    """Verify mental/emotional queries route to mental_health, not symptom."""

    @pytest.mark.parametrize("query", [
        "I feel so anxious lately",
        "I've been feeling depressed for weeks",
        "I'm having panic attacks",
        "I feel hopeless and sad",
        "I'm overwhelmed with stress",
        "can you help with my anxiety",
        "I think I need therapy",
        "I'm struggling with grief",
        "I feel so lonely all the time",
    ])
    def test_mental_health_queries(self, query):
        assert classify_intent(query) == "mental_health"

    def test_mental_health_beats_symptom(self):
        """Emotional keywords should route to mental_health, not symptom."""
        assert classify_intent("I feel depressed and anxious") == "mental_health"

    def test_mental_health_does_not_beat_urgent(self):
        """Suicidal intent should still be urgent, not mental_health."""
        assert classify_intent("I'm depressed and want to kill myself") == "urgent"


# ── Bug 3: false-positive prevention (word-boundary matching) ────────────────

class TestFalsePositivePrevention:
    """
    Verify that short keywords like 'ast', 'alt', 'gel', 'gas', 'cold',
    'test', 'period' do NOT false-positive match inside ordinary words.
    """

    @pytest.mark.parametrize("query,must_not_be", [
        # "ast" inside "breakfast" / "fast"
        ("let's grab breakfast fast", "lab"),
        ("that was a fast ride", "lab"),
        # "alt" inside "salt" / "alternative"
        ("pass the salt please", "lab"),
        ("what's an alternative route", "lab"),
        # "gel" inside "angel"
        ("you're an angel", "prescription"),
        # "gas" inside "Vegas"
        ("I'm going to Vegas", "symptom"),
        # "cold" inside "scold"
        ("don't scold me", "general"),
        # "test" inside "contest" / "attest"
        ("I won the contest", "lab"),
        ("I can attest to that", "lab"),
    ])
    def test_substring_does_not_match(self, query, must_not_be):
        result = classify_intent(query)
        assert result != must_not_be, (
            f"'{query}' was classified as '{result}' — "
            f"false positive for '{must_not_be}'"
        )

    def test_legitimate_lab_keywords_still_match(self):
        """Whole-word 'ast' and 'alt' should still match for lab intent."""
        assert classify_intent("my ast level is high") == "lab"
        assert classify_intent("what does alt mean in blood test") == "lab"

    def test_legitimate_symptom_keywords_still_match(self):
        """Whole-word 'gas' should still match for symptom intent."""
        assert classify_intent("I have a lot of gas and bloating") == "symptom"

    def test_legitimate_prescription_keywords_still_match(self):
        """Whole-word 'gel' should still match for prescription intent."""
        assert classify_intent("the doctor prescribed a gel") == "prescription"


# ── Standard classification ──────────────────────────────────────────────────

class TestStandardClassification:
    """Verify the basic intent routing still works correctly."""

    def test_lab_intent(self):
        assert classify_intent("what do my blood test results mean") == "lab"
        assert classify_intent("my hemoglobin is low") == "lab"

    def test_prescription_intent(self):
        assert classify_intent("what is this medicine prescribed for") == "prescription"
        assert classify_intent("tell me about paracetamol dosage") == "prescription"

    def test_symptom_intent(self):
        assert classify_intent("I have a headache and fever") == "symptom"
        assert classify_intent("my stomach hurts after eating") == "symptom"

    def test_general_intent(self):
        assert classify_intent("what is diabetes") == "general"
        assert classify_intent("how does the liver work") == "general"

    def test_greeting(self):
        assert classify_intent("hello") == "greeting"
        assert classify_intent("hey there") == "greeting"

    def test_farewell(self):
        assert classify_intent("goodbye") == "farewell"
        assert classify_intent("take care") == "farewell"

    def test_off_topic(self):
        assert classify_intent("what is the capital of France") == "off_topic"
        assert classify_intent("tell me a joke") == "off_topic"


# ── is_health_related uses word-boundary matching ────────────────────────────

class TestIsHealthRelated:
    def test_health_related_positive(self):
        assert is_health_related("I have diabetes") is True
        assert is_health_related("what is blood pressure") is True

    def test_health_related_negative(self):
        assert is_health_related("what is the weather today") is False

    def test_no_false_positive_from_substring(self):
        """'test' inside 'contest' should not trigger health-related."""
        assert is_health_related("I won the contest easily") is False
