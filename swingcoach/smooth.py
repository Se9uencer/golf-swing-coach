"""Low-pass filtering of landmark traces. Pure numpy/scipy, no I/O."""

import numpy as np
from scipy.signal import butter, filtfilt


def _interpolate_nan_1d(trace: np.ndarray) -> np.ndarray:
    """Linear interpolation across NaN gaps; holds the edge value beyond
    the first/last valid sample rather than extrapolating."""
    valid = ~np.isnan(trace)
    if valid.sum() < 2:
        # Nothing usable to interpolate from — leave as-is; the caller's
        # confidence tracking is what should flag this, not a fabricated fill.
        return trace
    idx = np.arange(len(trace))
    out = trace.copy()
    out[~valid] = np.interp(idx[~valid], idx[valid], trace[valid])
    return out


def lowpass(traces: np.ndarray, fps: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass over the first axis (time).

    `traces` may be any shape `(n_frames, ...)`. NaN gaps (frames where a
    landmark wasn't detected) are linearly interpolated before filtering —
    filtfilt does not handle NaN, and differentiating raw gaps is worse than
    interpolating across them.

    Always filtfilt, never lfilter: lfilter introduces phase lag, which
    shifts every detected event by a variable, hard-to-notice amount. See
    MISTAKES.md.
    """
    nyquist = fps / 2.0
    if not (0 < cutoff_hz < nyquist):
        raise ValueError(f"cutoff_hz={cutoff_hz} must be within (0, {nyquist}) at fps={fps}")

    n_frames = traces.shape[0]
    min_len = 3 * (order + 1)  # filtfilt's default pad requirement, roughly
    if n_frames < min_len:
        raise ValueError(
            f"clip too short to filter: {n_frames} frames, need >= {min_len} for order={order}"
        )

    b, a = butter(order, cutoff_hz / nyquist, btype="low")

    flat = traces.reshape(n_frames, -1)
    out = np.empty_like(flat, dtype=np.float64)
    for col in range(flat.shape[1]):
        filled = _interpolate_nan_1d(flat[:, col].astype(np.float64))
        out[:, col] = filtfilt(b, a, filled)

    return out.reshape(traces.shape).astype(np.float32)
