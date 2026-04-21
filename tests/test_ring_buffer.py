"""Tests for ring buffer."""

import numpy as np

from ccxsa.utils.ring_buffer import RingBuffer


def test_write_and_read():
    buf = RingBuffer(capacity_frames=1000, channels=1)
    data = np.arange(100, dtype=np.int16)
    buf.write(data)
    assert buf.available() == 100
    result = buf.read(100)
    assert result is not None
    np.testing.assert_array_equal(result, data)


def test_read_empty():
    buf = RingBuffer(capacity_frames=1000, channels=1)
    assert buf.read(100) is None


def test_wraparound():
    buf = RingBuffer(capacity_frames=100, channels=1)
    # Fill most of the buffer
    data1 = np.arange(80, dtype=np.int16)
    buf.write(data1)
    buf.read(80)
    # Write more data that wraps around
    data2 = np.arange(50, dtype=np.int16)
    buf.write(data2)
    result = buf.read(50)
    assert result is not None
    np.testing.assert_array_equal(result, data2)


def test_overflow_clamps():
    buf = RingBuffer(capacity_frames=10, channels=1)
    data = np.arange(20, dtype=np.int16)
    written = buf.write(data)
    assert written == 10


def test_wakeword_detector_exact():
    from ccxsa.wakeword.detector import WakeWordDetector

    det = WakeWordDetector(phrases=["チチクサ"])
    assert det.check("チチクサ")
    assert det.check("ちちくさ")
    assert det.check("えーとチチクサお願い")
    assert not det.check("こんにちは")


def test_wakeword_detector_fuzzy():
    from ccxsa.wakeword.detector import WakeWordDetector

    det = WakeWordDetector(phrases=["チチクサ"], fuzzy_threshold=2)
    # Whisper may produce these variations
    assert det.check("ちくさ")  # 1 deletion
    assert det.check("ちちぐさ")  # 1 substitution
    assert det.check("チチクサー")  # 1 insertion (long vowel)
    assert not det.check("おはようございます")  # unrelated
