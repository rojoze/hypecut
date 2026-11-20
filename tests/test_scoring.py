import numpy as np

from hypecut.scoring import hype_curve, pick_moments, robust_z


def _curve_with_peaks(n=600, peaks=(100, 300, 500)):
    rng = np.random.default_rng(0)
    loud = rng.normal(-30, 1.5, n)
    for p in peaks:
        loud[p:p + 4] = -6
    spike = loud - np.median(loud)
    return loud, spike


def test_robust_z_ignores_outlier():
    x = np.array([1, 1, 1, 1, 100.0])
    z = robust_z(x)
    assert z[-1] > 10 and abs(z[0]) < 1e-6 or z[0] <= 0


def test_finds_all_peaks_and_no_overlap():
    loud, spike = _curve_with_peaks()
    curve, comps = hype_curve(loud, spike, None, None)
    ms = pick_moments(curve, comps, duration=600, top=5)
    peaks = sorted(m.peak for m in ms)
    assert len(ms) == 3
    for want, got in zip((100, 300, 500), peaks):
        assert abs(got - want) <= 4
    spans = sorted((m.start, m.end) for m in ms)
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a1 <= b0


def test_lengths_respected():
    loud, spike = _curve_with_peaks()
    curve, comps = hype_curve(loud, spike, None, None)
    for m in pick_moments(curve, comps, duration=600, top=5, min_len=20, max_len=30):
        assert 20 - 1e-6 <= m.duration <= 30 + 1e-6
        assert m.start <= m.peak <= m.end


def test_peak_at_start_clamped():
    loud, spike = _curve_with_peaks(peaks=(2,))
    curve, comps = hype_curve(loud, spike, None, None)
    ms = pick_moments(curve, comps, duration=600, top=1, min_len=15)
    assert ms and ms[0].start == 0 and ms[0].duration >= 15


def test_quiet_vod_returns_nothing():
    loud = np.full(300, -30.0)
    curve, comps = hype_curve(loud, np.zeros(300), None, None)
    assert pick_moments(curve, comps, duration=300) == []


def test_chat_shifts_ranking():
    n = 600
    loud = np.random.default_rng(1).normal(-30, 1, n)
    loud[100:104] = -12   # loud but chat silent
    loud[400:404] = -14   # slightly quieter but chat explodes
    chat = np.zeros(n)
    chat[398:410] = 40
    curve, comps = hype_curve(loud, loud - np.median(loud), chat, None)
    ms = pick_moments(curve, comps, duration=n, top=2)
    assert abs(ms[0].peak - 402) <= 8
