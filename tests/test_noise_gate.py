"""Tests for noise gate."""

import numpy as np

from ccxsa.audio.noise_gate import NoiseGate


def test_noise_gate_silences_quiet_audio():
    gate = NoiseGate(sample_rate=16000, gate_threshold_db=-30.0)
    # Very quiet audio: all near zero
    quiet = np.zeros(480, dtype=np.int16)
    result = gate.process(quiet)
    assert np.all(result == 0)


def test_noise_gate_passes_loud_audio():
    gate = NoiseGate(sample_rate=16000, gate_threshold_db=-60.0)
    # Generate a 1kHz sine wave at moderate volume
    t = np.arange(480) / 16000.0
    signal = (np.sin(2 * np.pi * 1000 * t) * 16000).astype(np.int16)
    result = gate.process(signal)
    # Should not be all zeros
    assert not np.all(result == 0)


def test_noise_gate_bandpass():
    gate = NoiseGate(sample_rate=16000, low_cut=200.0, high_cut=4000.0)
    # Very low frequency (10 Hz) should be filtered out
    t = np.arange(480) / 16000.0
    low_freq = (np.sin(2 * np.pi * 10 * t) * 20000).astype(np.int16)
    result = gate.process(low_freq)
    # Energy should be significantly reduced
    assert np.sqrt(np.mean(result.astype(float) ** 2)) < np.sqrt(
        np.mean(low_freq.astype(float) ** 2)
    )
