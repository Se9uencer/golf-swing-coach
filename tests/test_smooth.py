"""Unit tests for smooth.py's math. Verifies the filter behaves as designed
on synthetic signals -- this is NOT a substitute for checking a real overlay
render, per AGENTS.md. It answers "is the Butterworth/filtfilt call correct",
not "does this work on a real swing"."""

import numpy as np
import pytest

from swingcoach.smooth import lowpass

FPS = 240.0


def _sine(freq_hz: float, n: int, fps: float = FPS) -> np.ndarray:
    t = np.arange(n) / fps
    return np.sin(2 * np.pi * freq_hz * t)


def test_attenuates_high_frequency():
    # A wrist-jitter-like 60Hz wiggle should be almost entirely removed
    # by a 12Hz cutoff.
    n = 240
    noisy = _sine(60.0, n)
    filtered = lowpass(noisy, fps=FPS, cutoff_hz=12.0)
    # ignore edge transients from filtfilt's padding
    core = slice(30, n - 30)
    assert np.std(filtered[core]) < 0.1 * np.std(noisy[core])


def test_preserves_low_frequency():
    # A real swing-scale motion (~2Hz) should pass through with amplitude
    # close to 1 and no meaningful shift -- proves filtfilt (zero-phase),
    # not lfilter, is actually being used.
    n = 240
    slow = _sine(2.0, n)
    filtered = lowpass(slow, fps=FPS, cutoff_hz=12.0)
    core = slice(30, n - 30)
    correlation = np.corrcoef(slow[core], filtered[core])[0, 1]
    assert correlation > 0.99


def test_interpolates_nan_gaps():
    n = 240
    trace = _sine(2.0, n)
    with_gap = trace.copy()
    with_gap[100:105] = np.nan
    filtered = lowpass(with_gap, fps=FPS, cutoff_hz=12.0)
    assert not np.isnan(filtered).any()


def test_rejects_cutoff_above_nyquist():
    trace = np.zeros(240)
    with pytest.raises(ValueError):
        lowpass(trace, fps=FPS, cutoff_hz=200.0)


def test_rejects_clip_too_short():
    trace = np.zeros(5)
    with pytest.raises(ValueError):
        lowpass(trace, fps=FPS, cutoff_hz=12.0)


def test_operates_on_landmark_shaped_arrays():
    # Real usage: (n_frames, 33, 2)
    n = 240
    traces = np.stack([_sine(2.0, n)] * 33 * 2, axis=-1).reshape(n, 33, 2)
    filtered = lowpass(traces, fps=FPS, cutoff_hz=12.0)
    assert filtered.shape == traces.shape
