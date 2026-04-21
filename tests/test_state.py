"""Tests for state machine."""

from ccxsa.state import AppState


def test_state_values():
    assert AppState.IDLE.value == "idle"
    assert AppState.LISTENING.value == "listening"
    assert AppState.PROCESSING.value == "processing"
    assert AppState.SPEAKING.value == "speaking"


def test_state_members():
    assert len(AppState) == 4
