import numpy as np

from rockfall.detect.features import aggregate, compute_features

FS = 200


def test_sine_is_narrowband_with_correct_frequency() -> None:
    t = np.arange(10 * FS) / FS
    f = compute_features(np.sin(2 * np.pi * 12.0 * t), FS, 0.0, 10.0)
    assert f is not None
    assert abs(f.dom_freq_hz - 12.0) < 1.0 and f.peakiness > 30 and f.crest_factor < 1.6 and f.env_cv < 0.2


def test_impulses_have_high_crest_and_burstiness() -> None:
    x = np.random.default_rng(0).normal(0, 0.1, 10 * FS)
    x[[400, 1100, 1500]] = 20.0
    f = compute_features(x, FS, 0.0, 10.0)
    assert f is not None and f.crest_factor > 10 and f.env_cv > 0.8


def test_rise_time_of_ramp() -> None:
    n = 10 * FS
    x = np.random.default_rng(1).normal(size=n) * np.linspace(0.05, 1.0, n)
    f = compute_features(x, FS, 0.0, 10.0)
    assert f is not None and f.rise_time_s > 7.0


def test_short_window_returns_none_and_duration_uses_window() -> None:
    x = np.random.default_rng(0).normal(size=5 * FS)
    assert compute_features(x, FS, 0.0, 0.5) is None
    f = compute_features(x, FS, 1.0, 4.0, ongoing=True)
    assert f is not None and f.duration_s == 3.0 and f.ongoing


def test_aggregate_uses_max_for_size_and_median_for_shape() -> None:
    t = np.arange(4 * FS) / FS
    parts = [compute_features(a * np.sin(2 * np.pi * f0 * t) + 0.01, FS, 0.0, 4.0) for a, f0 in ((1, 10), (5, 10), (2, 40))]
    agg = aggregate([p for p in parts if p])
    assert agg["peak_amp"] > 4.9 and abs(agg["dom_freq_hz"] - 10) < 1.0
    assert aggregate([]) == {}
