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

def test_radar_parametric_eq_response(radar_tuning):
    """+5dB low shelf at 150Hz and +2dB peak at 80Hz (Q .9), in stock's own
    design (libasp.so 0x932e8). Its shelf sits 3dB under its boost at the
    corner (Zoelzer), not half of it as the cookbook's does, so 150Hz reads
    +3.2dB from the shelf and +4.1dB with the peak's skirt. Values are what
    the decoded formulas give; the top of the band is untouched."""
    from scipy.signal import sosfreqz
    fs = 48000
    sos = em_eq.radar_peq_sos(fs, radar_tuning.peq)
    f = np.array([20.0, 80.0, 150.0, 300.0, 5000.0])
    _, h = sosfreqz(sos, worN=f, fs=fs)
    db = 20 * np.log10(np.abs(h))
    assert 5.1 < db[0] < 5.3      # the shelf's 5dB, peak skirt negligible
    assert 6.7 < db[1] < 6.85     # shelf plus the peak's 2dB
    assert 4.0 < db[2] < 4.2      # at the shelf corner
    assert 0.7 < db[3] < 0.8      # stock's shelf is still lifting here
    assert abs(db[4]) < 0.1


def test_radar_parametric_shelf_ignores_q_as_stock_does():
    """libasp's LOW_SHELF design never reads Q (case 5 uses a fixed sqrt(2)K
    term); ParametricEQ.cfg's 'Q 0.9' on the shelf is therefore not an
    input. The peak does read it."""
    import inspect
    assert "Q" not in inspect.signature(em_eq._stock_loshelf_sos).parameters
    a = em_eq._stock_peak_sos(80.0, 2.0, 0.9, 48000)
    b = em_eq._stock_peak_sos(80.0, 2.0, 2.0, 48000)
    assert not np.allclose(a, b)


def test_radar_stock_curve_applies_output_trim(radar_tuning):
    """A flat-ish stream through the stock curve ends 3dB hotter than the
    trim alone would explain only if the trim is applied: compare against the
    same chain with the trim removed."""
    fs = 48000
    t = np.arange(fs // 2) / fs
    x = (4000 * np.sin(2 * np.pi * 2000 * t)).astype(np.int16).tobytes()
    a = em_eq.StreamingEQ(fs, stock_curve=True)
    b = em_eq.StreamingEQ(fs, stock_curve=True)
    b._trim = 1.0
    ya = np.frombuffer(a.process(x), np.int16).astype(float)[fs // 4:]
    yb = np.frombuffer(b.process(x), np.int16).astype(float)[fs // 4:]
    ratio_db = 20 * np.log10(np.sqrt((ya**2).mean()) / np.sqrt((yb**2).mean()))
    assert abs(ratio_db - radar_tuning.trim_db) < 0.05


# ─── Radar's volume-banded stock FIR (AFE.cfg "Volume Boundary") ─────────────

def _level_gain(level):
    return 0.0 if level == 0 else 10 ** ((level - 127) / 40)


@pytest.mark.parametrize("level,value,band", [
    (0, 0, 0), (3, 1, 0), (37, 10, 0), (38, 11, 0), (77, 50, 0), (78, 51, 1),
    (87, 60, 1), (88, 61, 2), (97, 70, 2), (98, 71, 3), (107, 80, 3),
    (108, 81, 4), (127, 100, 4),
])
def test_radar_eq_band_by_volume(level, value, band):
    """Pinned against the same table as the Go test, so the two ends cannot
    pick different curves at the same volume. A level on a boundary takes
    that boundary's own file, as libasp does."""
    g = _level_gain(level)
    assert em_eq.stock_volume_value(g) == value
    assert em_eq.radar_eq_band(g, [50, 60, 70, 80, 100]) == band


def test_stock_mixer_levels_are_what_bin_mixer_holds():
    """The table read out of stock's /system/bin/mixer: 101 levels, 0.5dB
    per step, 127 = 0dB, value + 27 from value 11 up. It reproduces
    StandAloneAAModules.cfg's music_volTab at every Alexa step."""
    t = em_eq.STOCK_MIXER_LEVELS
    assert t[:11] == (0, 3, 7, 11, 17, 20, 27, 30, 32, 35, 36)
    assert all(t[v] == v + 27 for v in range(11, 101))
    volume_curve = [0, 1, 2, 3, 4, 6, 11, 16, 22, 28, 34, 40, 44, 46, 50, 54,
                    56, 60, 64, 68, 70, 72, 76, 80, 84, 88, 90, 92, 96, 98, 100]
    music_voltab = [-62.0, -60.2, -58.1, -55.0, -50.1, -44.5, -42.0, -39.0,
                    -36.0, -33.0, -30.0, -28.0, -27.0, -25.0, -23.0, -22.0,
                    -20.0, -18.0, -16.0, -15.0, -14.0, -12.0, -10.0, -8.0,
                    -6.0, -5.0, -4.0, -2.0, -1.0, 0.0]
    # Steps 1..30. The mixer's levels are whole 0.5dB steps, so the table's
    # fractional low end (-60.2, -58.1, -50.1) agrees to within one step.
    for step in range(1, 31):
        db = (t[volume_curve[step]] - 127) / 2
        assert abs(db - music_voltab[step - 1]) <= 1.0, step


def test_stock_volume_value_is_level_minus_27_from_38_up():
    for level in range(38, 128):
        assert em_eq.stock_volume_value(_level_gain(level)) == level - 27


def test_radar_banded_files_are_a_loudness_compensation(radar_tuning):
    """The point of selecting by volume: the bass boost backs off as the
    volume goes up. If a future extraction ever produced five copies of one
    curve, this is where it would show."""
    bounds, taps = em_eq._radar_eq_banded()
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


def test_chain_without_a_volume_keeps_eq50(radar_tuning):
    """Every caller that does not pass volume_gain (the controller-side
    chain) keeps the single EQ_50 curve it always had."""
    eq = em_eq.StreamingEQ(48000, stock_curve=True)
    assert eq._fir_bounds is None and len(eq._fir._hs) == 1


def test_chain_that_takes_the_volume_rounds_to_s16():
    """Radar's volume sits ahead of the stages, so the lowest steps leave a few
    LSB of signal; truncating toward zero lost ~6dB of it and zeroed anything
    under 1 LSB. The cast rounds (half to even) there and truncates elsewhere."""
    x = np.array([0.4, 0.6, 1.5, 2.5, -0.6, -1.6, 40000.0, -40000.0])
    r = np.frombuffer(em_eq._to_int16(x, True), np.int16)
    t = np.frombuffer(em_eq._to_int16(x, False), np.int16)
    assert list(r) == [0, 1, 2, 2, -1, -2, 32767, -32768]
    assert list(t) == [0, 0, 1, 2, 0, -1, 32767, -32768]
