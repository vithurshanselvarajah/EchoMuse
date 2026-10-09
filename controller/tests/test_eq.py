import math
import pytest

import numpy as np

import em_eq

RATE = 48000


def _sine(freq: float, seconds: float = 0.5, amp: float = 0.25) -> bytes:
    t = np.arange(int(RATE * seconds)) / RATE
    pcm = (np.sin(2 * math.pi * freq * t) * amp * 32767).astype(np.int16)
    return pcm.tobytes()


def _rms(pcm: bytes) -> float:
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    return float(np.sqrt(np.mean(x * x)))


def test_flat_bands_are_transparent():
    pcm = _sine(1000)
    out = em_eq.apply(pcm, RATE, bands=[0.0] * 8)
    assert len(out) == len(pcm)
    # 0 dB everywhere should be within a fraction of a dB of identity.
    ratio = _rms(out) / _rms(pcm)
    assert 0.97 < ratio < 1.03


def test_none_bands_default_to_flat():
    pcm = _sine(1000)
    assert abs(_rms(em_eq.apply(pcm, RATE)) - _rms(pcm)) / _rms(pcm) < 0.03


def test_band_boost_raises_its_own_frequency_only():
    # 60 Hz sits below the 125 Hz shelf corner (full +6 dB); at the corner
    # itself a shelf only delivers half its gain.
    low = _sine(60)
    high = _sine(8000)
    bands = [6.0, 0, 0, 0, 0, 0, 0, 0]  # +6 dB low shelf
    low_gain = _rms(em_eq.apply(low, RATE, bands=bands)) / _rms(low)
    high_gain = _rms(em_eq.apply(high, RATE, bands=bands)) / _rms(high)
    assert low_gain > 1.7          # ~+6 dB ≈ ×2
    assert 0.9 < high_gain < 1.1   # shelf must not leak into the top band


def test_cut_reduces_level():
    pcm = _sine(1000)
    out = em_eq.apply(pcm, RATE, bands=[0, 0, 0, -12.0, 0, 0, 0, 0])
    assert _rms(out) / _rms(pcm) < 0.5


def test_short_and_empty_input_pass_through():
    assert em_eq.apply(b"", RATE) == b""
    assert em_eq.apply(b"\x01", RATE) == b"\x01"


def test_wrong_band_count_still_returns_audio():
    pcm = _sine(1000)
    out = em_eq.apply(pcm, RATE, bands=[0.0, 0.0])  # padded internally
    assert len(out) == len(pcm)


def test_streaming_eq_matches_batch_apply():
    pcm = _sine(1000, seconds=0.4)
    bands = [3.0, 0, -2.0, 0, 0, 4.0, 0, 1.0]
    want = em_eq.apply(pcm, RATE, bands=bands)
    eq = em_eq.StreamingEQ(RATE, bands=bands)
    out = b""
    for i in range(0, len(pcm), 4096):
        out += eq.process(pcm[i:i + 4096])
    # Filter state carries across chunks — output must match the
    # whole-buffer path (same filters, same float32 pipeline).
    got = np.frombuffer(out, dtype=np.int16).astype(np.int32)
    ref = np.frombuffer(want, dtype=np.int16).astype(np.int32)
    assert np.abs(got - ref).max() <= 1  # ±1 LSB float rounding


def test_streaming_eq_flat_is_passthrough():
    eq = em_eq.StreamingEQ(RATE, bands=[0.0] * 8)
    chunk = _sine(500, seconds=0.05)
    assert eq.process(chunk) == chunk


# ─── Radar ParametricEQ + OutputTrim (ParametricEQ.cfg / AFE.cfg) ─────────────

def test_radar_parametric_eq_response():
    """+5dB low shelf at 150Hz (Q .9) and +2dB peak at 80Hz (Q .9): the
    shelf's own gain at its corner is half (+2.5dB), the peak adds on top of
    it at 80Hz, and the top of the band is untouched."""
    from scipy.signal import sosfreqz
    fs = 48000
    sos = em_eq.radar_peq_sos(fs)
    f = np.array([20.0, 80.0, 150.0, 5000.0])
    _, h = sosfreqz(sos, worN=f, fs=fs)
    db = 20 * np.log10(np.abs(h))
    assert 4.8 < db[0] < 5.6      # the shelf's 5dB, peak skirt negligible
    assert 6.8 < db[1] < 7.5    # shelf plus the peak's 2dB
    assert 2.8 < db[2] < 3.8      # near the shelf corner
    assert abs(db[3]) < 0.1


def test_radar_stock_curve_applies_output_trim():
    """A flat-ish stream through the stock curve ends 3dB hotter than the
    trim alone would explain only if the trim is applied: compare against the
    same chain with the trim removed."""
    fs = 48000
    t = np.arange(fs // 2) / fs
    x = (4000 * np.sin(2 * np.pi * 2000 * t)).astype(np.int16).tobytes()
    a = em_eq.StreamingEQ(fs, stock_curve=True)
    if a._fir is None:
        pytest.skip("radar_eq_taps.json not present")
    b = em_eq.StreamingEQ(fs, stock_curve=True)
    b._trim = 1.0
    ya = np.frombuffer(a.process(x), np.int16).astype(float)[fs // 4:]
    yb = np.frombuffer(b.process(x), np.int16).astype(float)[fs // 4:]
    ratio_db = 20 * np.log10(np.sqrt((ya**2).mean()) / np.sqrt((yb**2).mean()))
    assert abs(ratio_db - em_eq.RADAR_OUTPUT_TRIM_DB) < 0.05


# ─── Radar's volume-banded stock FIR (AFE.cfg "Volume Boundary") ─────────────

def _level_gain(level):
    return 0.0 if level == 0 else 10 ** ((level - 127) / 40)


@pytest.mark.parametrize("level,band", [
    (0, 0), (47, 0), (81, 0), (82, 1), (93, 1), (94, 2),
    (101, 2), (102, 3), (110, 3), (111, 4), (127, 4),
])
def test_radar_eq_band_by_volume(level, band):
    """Pinned against the same table as the Go test, so the two ends cannot
    pick different curves at the same volume. Levels 93 and 110 land exactly
    on index 60 and 80 and must take that boundary's own file."""
    assert em_eq.radar_eq_band(_level_gain(level), [50, 60, 70, 80, 100]) == band


def test_stock_volume_index_follows_the_speaker_curve():
    # The curve's own points come back as themselves.
    for idx, att in em_eq.SPEAKER_MUSIC_CURVE:
        assert em_eq.stock_volume_index(10 ** (att / 20)) == pytest.approx(idx)
    assert em_eq.stock_volume_index(0.0) == 0.0
    assert em_eq.stock_volume_index(1e-6) == 1.0     # below the curve: index 1
    assert em_eq.stock_volume_index(2.0) == 100.0    # above unity: index 100


def test_radar_banded_files_are_a_loudness_compensation():
    """The point of selecting by volume: the bass boost backs off as the
    volume goes up. If a future extraction ever produced five copies of one
    curve, this is where it would show."""
    banded = em_eq._radar_eq_banded()
    if banded is None:
        pytest.skip("radar_eq_banded.json not present")
    bounds, taps = banded
    assert bounds == [50, 60, 70, 80, 100]
    boost80 = [20 * np.log10(abs(np.fft.rfft(t, 48000)[80])) for t in taps]
    assert all(a > b for a, b in zip(boost80, boost80[1:])), boost80
    assert boost80[0] > 9.0 and boost80[-1] < 2.0


def test_banded_fir_crossfades_on_a_switch():
    a = np.zeros(32); a[0] = 1.0
    b = np.zeros(32); b[0] = 0.25
    f = em_eq._OverlapSaveFIR([a, b])
    x = np.full(256, 1000.0)
    f.process(x)
    f.set_band(1)
    y = f.process(x)
    assert y[0] == pytest.approx(1000 * (1 - 1 / 256) + 250 / 256)
    assert y[-1] == pytest.approx(250.0)
    assert np.allclose(f.process(x), 250.0)


def test_chain_without_a_volume_keeps_eq50():
    """Every caller that does not pass volume_gain (the controller-side
    chain) keeps the single EQ_50 curve it always had."""
    eq = em_eq.StreamingEQ(48000, stock_curve=True)
    if eq._fir is None:
        pytest.skip("radar_eq_taps.json not present")
    assert eq._fir_bounds is None and len(eq._fir._hs) == 1
