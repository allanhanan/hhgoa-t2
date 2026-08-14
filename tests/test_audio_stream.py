"""Unit tests for audio stream energy calculation and speech thresholding."""

from __future__ import annotations

import struct
import pytest
from voice_optimized_rag.voice.audio_stream import AudioStream


def test_calculate_energy_silence():
    audio = AudioStream()
    silence_frame = bytes(960)  # 480 samples of int16 zeros
    energy = audio.calculate_energy(silence_frame)
    assert energy == 0.0


def test_calculate_energy_low_signal():
    audio = AudioStream()
    # Create 480 samples with low amplitude int16 (max ~100)
    samples = [100 if i % 2 == 0 else -100 for i in range(480)]
    frame = struct.pack(f"<480h", *samples)
    energy = audio.calculate_energy(frame)
    assert energy == 100.0


def test_effective_threshold_dynamic_calibration():
    audio = AudioStream()
    # Ambient noise floor = 5.0
    threshold = audio.get_effective_threshold(5.0)
    # Expected: max(15.0, 5.0 * 3.0, 5.0 + 10.0) = 15.0
    assert threshold == 15.0

    # Ambient noise floor = 20.0
    threshold2 = audio.get_effective_threshold(20.0)
    # Expected: max(15.0, 60.0, 30.0) = 60.0
    assert threshold2 == 60.0


def test_effective_threshold_configured():
    audio = AudioStream(energy_threshold=25.0)
    assert audio.get_effective_threshold(5.0) == 25.0
    assert audio.get_effective_threshold(100.0) == 25.0
