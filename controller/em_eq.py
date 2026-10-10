"""
em_eq.py — Output EQ for EchoMuse Controller
=============================================

Applies a biquad filter chain to mono S16_LE PCM (Piper TTS output) before
it is resampled and streamed to the device speaker.

Eight independently controllable bands covering the Echo Dot Gen 2's useful
output range. Each band is a gain value in dB; 0.0 = flat (no effect).

Band centre frequencies and types:
  0:  125 Hz  — low shelf
  1:  250 Hz  — peaking, Q=1.4
  2:  500 Hz  — peaking, Q=1.4
  3: 1000 Hz  — peaking, Q=1.4
  4: 2000 Hz  — peaking, Q=1.4
  5: 3500 Hz  — peaking, Q=1.4
  6: 5500 Hz  — peaking, Q=1.4
  7: 8000 Hz  — high shelf

All filter design uses the Audio EQ Cookbook by Robert Bristow-Johnson.
High-pass uses scipy.signal.butter (already a dependency via openwakeword).

Usage:
    import em_eq
    eq_pcm = em_eq.apply(voice_response, SPEAKER_RATE, bands=[0]*8, loudness=False)
"""

import json
import math
import os
import logging
import numpy as np
from scipy.signal import sosfilt

import em_limiter
import em_volume
import em_mbc  # noqa: F401  (type reference in signatures)

log = logging.getLogger("echomuse.eq")

EQ_FREQUENCIES = [125, 250, 500, 1000, 2000, 3500, 5500, 8000]
NUM_BANDS       = len(EQ_FREQUENCIES)
DEFAULT_BANDS   = [0.0] * NUM_BANDS
_PEAK_Q         = 1.4   # ~1 octave bandwidth for middle bands


# ─── Biquad primitives ────────────────────────────────────────────────────────

def _peak_sos(fc: float, gain_db: float, Q: float, fs: float) -> np.ndarray:
    """Peaking parametric EQ biquad (Audio EQ Cookbook)."""
    A     = 10 ** (gain_db / 40.0)
    w0    = 2 * math.pi * fc / fs
    cw    = math.cos(w0)
    alpha = math.sin(w0) / (2 * Q)
    b0 = 1 + alpha * A;  b1 = -2 * cw;  b2 = 1 - alpha * A
    a0 = 1 + alpha / A;  a1 = -2 * cw;  a2 = 1 - alpha / A
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])


def _loshelf_sos(fc: float, gain_db: float, fs: float) -> np.ndarray:
    """Low shelf biquad (Audio EQ Cookbook, S=1)."""
    A     = 10 ** (gain_db / 40.0)
    w0    = 2 * math.pi * fc / fs
    cw    = math.cos(w0)
    sqA   = math.sqrt(A)
    alpha = math.sin(w0) / math.sqrt(2)   # S=1
    b0 =      A * ((A+1) - (A-1)*cw + 2*sqA*alpha)
    b1 =  2 * A * ((A-1) - (A+1)*cw)
    b2 =      A * ((A+1) - (A-1)*cw - 2*sqA*alpha)
    a0 =           (A+1) + (A-1)*cw + 2*sqA*alpha
    a1 =     -2 * ((A-1) + (A+1)*cw)
    a2 =           (A+1) + (A-1)*cw - 2*sqA*alpha
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])


def _hishelf_sos(fc: float, gain_db: float, fs: float) -> np.ndarray:
    """High shelf biquad (Audio EQ Cookbook, S=1)."""
    A     = 10 ** (gain_db / 40.0)
    w0    = 2 * math.pi * fc / fs
    cw    = math.cos(w0)
    sqA   = math.sqrt(A)
    alpha = math.sin(w0) / math.sqrt(2)   # S=1
    b0 =      A * ((A+1) + (A-1)*cw + 2*sqA*alpha)
    b1 = -2 * A * ((A-1) + (A+1)*cw)
    b2 =      A * ((A+1) + (A-1)*cw - 2*sqA*alpha)
    a0 =           (A+1) - (A-1)*cw + 2*sqA*alpha
    a1 =      2 * ((A-1) - (A+1)*cw)
    a2 =           (A+1) - (A-1)*cw - 2*sqA*alpha
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])


def _loshelf_q_sos(fc: float, gain_db: float, Q: float, fs: float) -> np.ndarray:
    """Low shelf biquad with an explicit Q (Audio EQ Cookbook,
    alpha = sin(w0)/(2Q)) — the form ParametricEQ.cfg states its shelf in.
    _loshelf_sos above is the S=1 special case, kept as it is so the 8-band
    EQ's vectors do not move."""
    A     = 10 ** (gain_db / 40.0)
    w0    = 2 * math.pi * fc / fs
    cw    = math.cos(w0)
    sqA   = math.sqrt(A)
    alpha = math.sin(w0) / (2 * Q)
    b0 =      A * ((A+1) - (A-1)*cw + 2*sqA*alpha)
    b1 =  2 * A * ((A-1) - (A+1)*cw)
    b2 =      A * ((A+1) - (A-1)*cw - 2*sqA*alpha)
    a0 =           (A+1) + (A-1)*cw + 2*sqA*alpha
    a1 =     -2 * ((A-1) + (A+1)*cw)
    a2 =           (A+1) + (A-1)*cw - 2*sqA*alpha
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])


def _loudness_sos(fs: float) -> np.ndarray:
    """Speech-range presence boost for lower listening volumes."""
    return _peak_sos(2500, 5.0, 0.8, fs)


# ─── Stock FIR curve (Radar only) ──────────────────────────────────────────────
#
# Radar's stock speaker EQ (EQ_50.cfg) is a 2048-tap FIR, not an 8-band
# parametric curve — a fundamentally different filter shape the 8 sliders
# above cannot reproduce (measured: the real curve swings from +10dB at
# 80Hz to -9dB at 200Hz, and -5dB at 2.5kHz to +3dB at 3.15kHz — both
# transitions narrower than a Q=1.4 band at any of the 8 fixed frequencies
# can track). This runs it directly via overlap-save rather than
# approximating it.
#
# Extracted from the owner's own Radar firmware (NS6572/6436), for that
# owner's personal build only — see JOURNAL/commit message for why this is
# not something to carry into a PR: Amazon's exact filter coefficients are
# not something this project otherwise redistributes.
# Radar's ParametricEQ.cfg ("EQv5.4") and OutputTrim, read off the unit's
# own /system/vendor/etc/audio-algorithms/ and AFE.cfg. AFE.cfg's
# Playback.Algorithms runs them as  EQ (FIR) -> ParametricEQ -> MBCL ->
# OutputTrim, so they ride the same stock_curve switch as the FIR: they are
# the same tuning, and the FIR alone is not what stock sounds like. Of the
# cfg's 8 biquads only the first two are not BYPASS. Both state Q=0.9.
RADAR_PEQ_LOW_SHELF = (150.0, 5.0, 0.9)    # Fc Hz, GaindB, Q
RADAR_PEQ_PEAK      = (80.0, 2.0, 0.9)
RADAR_OUTPUT_TRIM_DB = 3.0                 # flat gain after MBCL's limiter


def radar_peq_sos(fs: float) -> np.ndarray:
    return np.vstack([_loshelf_q_sos(*RADAR_PEQ_LOW_SHELF, fs),
                      _peak_sos(RADAR_PEQ_PEAK[0], RADAR_PEQ_PEAK[1],
                                RADAR_PEQ_PEAK[2], fs)])


_RADAR_EQ_TAPS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "radar_eq_taps.json")
_radar_eq_taps_cache: np.ndarray | None = None


def _radar_eq_taps() -> np.ndarray | None:
    """The Radar FIR's coefficients, loaded once. None if the data file
    is not present — this file is deliberately not required for every
    install, only for a build that wants the stock_curve option."""
    global _radar_eq_taps_cache
    if _radar_eq_taps_cache is None:
        try:
            with open(_RADAR_EQ_TAPS_PATH) as f:
                data = json.load(f)
            _radar_eq_taps_cache = np.asarray(data["taps"], dtype=np.float64)
        except FileNotFoundError:
            return None
    return _radar_eq_taps_cache


# Radar's stock FIR is not one curve: AFE.cfg's "Equalizer FIR" lists
# EQ_50/60/70/80/100.cfg against "Volume Boundary": [50,60,70,80,100], and
# they are five different curves — a loudness compensation, not one curve at
# five gains (that was biscuit's EQ files). EQ_50 boosts 80Hz by +10.1dB,
# EQ_80 by +5.4dB, EQ_100 by +1.4dB, so the bass boost backs off as the
# volume goes up and MBCL has less to hold down.
#
# The boundaries are on stock's 0-100 MUSIC VOLUME VALUE: libaudioCtrl maps
# each Alexa step to it (VolumeCurves.xml), the mixer daemon turns it into an
# attenuation and sends it to libasp as the Music volume, and libasp plays the
# first file whose boundary is >= it. The attenuation is STOCK_MIXER_LEVELS,
# read out of /system/bin/mixer (Mixer_AlgoRampGain): a level in the same law
# as ours — 0.5dB per step, 127 = 0dB — for each value 0..100. So the value
# for a volume is recovered by finding where our level falls in that table;
# from value 11 up the table is simply value + 27. (This replaced a mapping
# through Android's speaker volume curve, which stock does not use for Alexa
# audio at all — it put levels 78-81, 88-93, 98-101 and 108-110 one bassier
# file down.)
_RADAR_EQ_BANDED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "radar_eq_banded.json")
_radar_eq_banded_cache: tuple[list, list] | None = None

STOCK_MIXER_LEVELS = em_volume.STOCK_MIXER_LEVELS


def stock_volume_value(gain: float) -> int:
    """Stock's 0-100 music volume value for a linear volume gain: the
    highest value whose mixer level is at or below ours. A gain is always
    one of the device's own levels (0.5dB steps), so the level is recovered
    exactly; the 1e-6 absorbs the round trip through log10."""
    if gain <= 0.0:
        return 0
    level = 127.0 + 40.0 * math.log10(gain)
    value = 0
    for v, lv in enumerate(STOCK_MIXER_LEVELS):
        if lv <= level + 1e-6:
            value = v
    return value


def radar_eq_band(gain: float, boundaries) -> int:
    """Which of the banded FIRs stock plays at this volume gain: the first
    whose boundary is at or above the volume value (libasp's own rule)."""
    value = stock_volume_value(gain)
    for i, b in enumerate(boundaries):
        if value <= b:
            return i
    return len(boundaries) - 1


def _radar_eq_banded():
    """(boundaries, [taps, ...]) for Radar's volume-banded stock FIR, loaded
    once; None if the data file is not present."""
    global _radar_eq_banded_cache
    if _radar_eq_banded_cache is None:
        try:
            with open(_RADAR_EQ_BANDED_PATH) as f:
                data = json.load(f)
        except FileNotFoundError:
            return None
        _radar_eq_banded_cache = (list(data["boundaries"]),
                                  [np.asarray(t, dtype=np.float64)
                                   for t in data["taps"]])
    return _radar_eq_banded_cache


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


class _OverlapSaveFIR:
    """
    Streaming FIR convolution via overlap-save, FFT-based.

    Unlike StreamingEQ's biquads (which carry state sample-by-sample and
    accept any chunk size for free), a direct per-sample FIR convolution of
    a filter this long (2048 taps) costs O(chunk_len * 2048) — about 100x
    the cost of the whole existing 8-biquad chain, measured against the
    device's own budget in device/CLAUDE.md. FFT-based overlap-save turns
    that into O(N log N) per chunk, N being the FFT size (next_pow2 of the
    chunk plus the filter's own history), making it cheap enough to run per
    period on hardware built for 13 biquads, not a 2048-tap filter.

    Output length always equals input length, per call, matching
    StreamingEQ.process's contract — there is no internal buffering, no
    accumulated latency beyond the filter's own fixed group delay (the same
    group delay direct convolution would have; FFT changes only HOW it is
    computed, not what it computes), and no silent gaps or bursts in what a
    caller gets back.

    The overlap is the last (M-1) RAW INPUT samples seen, carried across
    calls regardless of how each call's chunk size compares to M — a call
    shorter than M-1 is handled the same way as one much longer than it.
    """

    def __init__(self, taps):
        # One filter, or several of equal length to switch between
        # (set_band). The input history is the signal's, not any filter's,
        # so it carries straight across a switch.
        bands = taps if isinstance(taps, (list, tuple)) else [taps]
        self._hs = [np.asarray(t, dtype=np.float64) for t in bands]
        self._m = self._hs[0].size
        if any(h.size != self._m for h in self._hs):
            raise ValueError("banded FIR taps must all be the same length")
        self._h = self._hs[0]
        self._overlap = np.zeros(self._m - 1, dtype=np.float64)
        self._h_fft_cache: dict[tuple[int, int], np.ndarray] = {}
        self._band = 0
        self._fade_from: int | None = None

    def _h_fft(self, n: int, band: int | None = None) -> np.ndarray:
        band = self._band if band is None else band
        v = self._h_fft_cache.get((band, n))
        if v is None:
            # Cached per filter and distinct FFT size seen — the filters
            # themselves never change, so a transform is only recomputed
            # when a caller's chunk length changes the required FFT size.
            v = np.fft.rfft(self._hs[band], n=n)
            self._h_fft_cache[(band, n)] = v
        return v

    def set_band(self, band: int) -> None:
        """Switch filter. The next process() call crossfades linearly from
        the old filter's output to the new one's across its samples: both
        filter the same input history, so this is a change of curve with no
        discontinuity in the signal, and the fade keeps the change of curve
        itself from landing as a step."""
        if band != self._band:
            self._fade_from = self._band
            self._band = band

    def reset(self) -> None:
        """Zero the carried history — for a mode switch, not a parameter
        change: this filter has no tunable parameters to carry state
        through."""
        self._overlap[:] = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        ext = np.concatenate([self._overlap, x])
        n = _next_pow2(ext.size)
        spec = np.fft.rfft(ext, n=n)
        y = np.fft.irfft(spec * self._h_fft(n), n=n)
        if self._fade_from is not None:
            y_old = np.fft.irfft(spec * self._h_fft(n, self._fade_from), n=n)
            w = np.arange(1, x.size + 1, dtype=np.float64) / x.size
            seg = slice(self._m - 1, self._m - 1 + x.size)
            y = y.copy()
            y[seg] = y_old[seg] * (1.0 - w) + y[seg] * w
            self._fade_from = None
        # The first (m-1) samples of a length-n circular convolution of an
        # (m-1+L)-sample signal against an m-tap filter are corrupted by
        # wraparound; the next L are the exact linear-convolution result for
        # this call's new samples (see commit message/JOURNAL for the proof
        # — it holds for any n >= len(ext), not only n == len(ext), which is
        # what lets this use a cheap next_pow2 rather than a tight bound).
        out = y[self._m - 1: self._m - 1 + x.size]
        self._overlap = ext[-(self._m - 1):].copy()
        return out


# ─── Public API ───────────────────────────────────────────────────────────────

def build_sos(bands: list, sample_rate: int, loudness: bool = False) -> np.ndarray:
    """
    Build a stacked SOS matrix for the given band gains and sample rate.

    Exposed separately so callers can cache the matrix when bands haven't
    changed between calls.
    """
    sections = []
    for i, (fc, gain_db) in enumerate(zip(EQ_FREQUENCIES, bands)):
        if i == 0:
            sections.append(_loshelf_sos(fc, gain_db, sample_rate))
        elif i == NUM_BANDS - 1:
            sections.append(_hishelf_sos(fc, gain_db, sample_rate))
        else:
            sections.append(_peak_sos(fc, gain_db, _PEAK_Q, sample_rate))
    if loudness:
        sections.append(_loudness_sos(sample_rate))
    return np.vstack(sections)


def apply(
    pcm: bytes,
    sample_rate: int,
    bands: list | None = None,
    loudness: bool = False,
    limiter: "em_limiter.Limiter | None" = None,
    guard: "em_mbc.BassGuard | None" = None,
) -> bytes:
    """
    Apply EQ to mono S16_LE PCM. Returns mono S16_LE PCM at the same rate.

    Args:
        pcm:         Raw mono S16_LE PCM bytes (decoded TTS audio).
        sample_rate: Sample rate of pcm (SPEAKER_RATE = 48000 in the
                     playback pipeline since the 48k decode change).
        bands:       List of NUM_BANDS (8) gain values in dB. None = flat.
        loudness:    Add a +5dB speech-range presence boost if True.
        limiter:     Optional peak limiter applied AFTER the EQ, in float, so
                     nothing is quantised twice. Without one this function
                     hard-clips whatever the EQ boosted past full scale, which
                     is #231.

    Returns:
        EQ-processed mono S16_LE PCM bytes, same length as input.
    """
    if len(pcm) < 2:
        return pcm

    if bands is None:
        bands = DEFAULT_BANDS

    if len(bands) != NUM_BANDS:
        log.warning(f"[eq] Expected {NUM_BANDS} bands, got {len(bands)} — padding with zeros")
        bands = list(bands) + [0.0] * (NUM_BANDS - len(bands))

    flat = not loudness and all(b == 0.0 for b in bands)
    if flat and limiter is None and guard is None:
        return pcm

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    if not flat:
        samples = sosfilt(build_sos(bands, sample_rate, loudness), samples)
    # Order matters: the guard removes excursion the driver cannot deliver,
    # THEN the limiter catches what is left. Limiting first would spend gain
    # reduction on bass that is about to be thrown away, pulling down the
    # midrange for no reason.
    # A guard that delays (Radar's stock MBCL, its look-ahead) is fed its
    # latency in silence and the same count dropped from the front, so a
    # one-shot buffer comes back aligned and whole.
    lat = getattr(guard, "latency", 0) if guard is not None else 0
    if guard is not None:
        if lat:
            samples = guard.process(np.concatenate((samples, np.zeros(lat))))[lat:]
        else:
            samples = guard.process(samples)
    if limiter is not None:
        samples = np.concatenate([limiter.process(samples), limiter.flush()])
    # Backstop only. With a limiter attached this must never engage; without
    # one it is the historical behaviour, preserved so a caller that passes no
    # limiter is no worse off than before.
    return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()


class StreamingEQ:
    """
    Chunk-by-chunk EQ with filter state carried across calls — for audio
    that can't be processed as one buffer (music streams). apply() on
    independent chunks would reset the biquad states at every boundary
    and click; this is bit-identical to apply() over the concatenation.
    """

    def __init__(self, sample_rate: int, bands: list | None = None,
                 loudness: bool = False,
                 limiter: "em_limiter.Limiter | None" = None,
                 guard: "em_mbc.BassGuard | None" = None,
                 stock_curve: bool = False,
                 volume_gain: float | None = None):
        self._limiter = limiter
        self._guard = guard
        self._sample_rate = int(sample_rate)   # set_bands rebuilds against it

        # Structural, like limiter/guard above: fixed for this feed, not
        # something update() can flip mid-stream — a config change here
        # takes effect on the NEXT feed, the same way output_chain_on_device
        # already decides eq/Passthrough once per feed rather than live.
        # Additive with the 8-band EQ below, not a replacement for it: the
        # curve is Radar's stock tonal correction, bands are still free to
        # shape further on top of it.
        # A chain that takes the volume (volume_gain) plays the stock FIR
        # stock would at that volume — see _RADAR_EQ_BANDED_PATH. One that
        # does not has no volume to go by and keeps EQ_50, as before.
        self._fir_bounds = None
        banded = (_radar_eq_banded()
                  if stock_curve and volume_gain is not None else None)
        if banded is not None:
            self._fir_bounds, band_taps = banded
            self._fir = _OverlapSaveFIR(band_taps)
            self._fir.set_band(radar_eq_band(float(volume_gain),
                                             self._fir_bounds))
            self._fir._fade_from = None   # the first curve, not a change of one
        else:
            taps = _radar_eq_taps() if stock_curve else None
            if stock_curve and taps is None:
                log.warning("[eq] stock_curve requested but radar_eq_taps.json "
                            "is missing — falling back to the 8-band EQ alone")
            self._fir = _OverlapSaveFIR(taps) if taps is not None else None
        # ParametricEQ and OutputTrim exist exactly when the FIR does (see
        # RADAR_PEQ_*): the device gates them on the same condition.
        if self._fir is not None:
            self._peq_sos = radar_peq_sos(self._sample_rate)
            self._peq_zi = np.zeros((self._peq_sos.shape[0], 2), dtype=np.float64)
            self._trim = 10 ** (RADAR_OUTPUT_TRIM_DB / 20.0)
        else:
            self._peq_sos = None
            self._trim = 1.0

        # Volume AHEAD of the chain (Radar), as stock does it: AudioFlinger
        # attenuates before the AFE's EQ/MBCL ever see the signal, so the
        # compressors engage only when the user has turned it up. None keeps
        # the old arrangement (volume applied after the chain, on the
        # device) for every caller that does not pass one. Ramped across
        # each process() call exactly as the device's softVolume ramps a
        # period — see set_volume_gain.
        self._vol_target = None if volume_gain is None else float(volume_gain)
        self._vol_cur = self._vol_target

        # Last values update() applied; None until it is first called, so the
        # first call always lands rather than matching a coincidental default.
        self._applied = None
        if bands is None:
            bands = DEFAULT_BANDS
        if len(bands) != NUM_BANDS:
            bands = list(bands) + [0.0] * (NUM_BANDS - len(bands))
        if not loudness and all(b == 0.0 for b in bands):
            self._sos = None  # flat — pure passthrough
        else:
            self._sos = build_sos(bands, sample_rate, loudness)
            self._zi  = np.zeros((self._sos.shape[0], 2), dtype=np.float64)

    @property
    def limiter(self):
        """The chain's limiter, for instrumentation. May be None."""
        return self._limiter

    @property
    def guard(self):
        """The chain's bass guard, for instrumentation. May be None."""
        return self._guard

    def update(self, *,
               bands: list | None = None,
               loudness: bool = False,
               limiter_enabled: bool | None = None,
               limiter_threshold: float | None = None,
               limiter_release: float | None = None,
               guard_enabled: bool | None = None,
               guard_db: float | None = None) -> bool:
        """
        Re-apply the whole chain's settings mid-stream. Returns True if
        anything moved.

        Called per chunk by the music feed, so it compares before it acts:
        the steady-state cost is one tuple comparison, and the processors are
        only touched when a value actually changed. That matters because the
        setters are cheap but rebuilding the EQ coefficients is not, and doing
        it 23 times a second for no reason would be silly.

        Everything here updates state IN PLACE. Nothing is reconstructed, so
        there is no discontinuity in the filter states, the limiter's held
        tail or the stream's latency — see the setters for why each of those
        would otherwise be audible.
        """
        wanted = (tuple(bands) if bands is not None else None, loudness,
                  limiter_enabled, limiter_threshold, limiter_release,
                  guard_enabled, guard_db)
        if wanted == self._applied:
            return False

        prev = self._applied
        self._applied = wanted

        # EQ only when the curve itself moved — the expensive branch.
        if prev is None or (prev[0], prev[1]) != (wanted[0], wanted[1]):
            self.set_bands(bands, loudness)

        if self._limiter is not None:
            self._limiter.set_params(threshold_db=limiter_threshold,
                                     release_ms=limiter_release,
                                     enabled=limiter_enabled)
        if self._guard is not None:
            self._guard.set_params(bass_guard_db=guard_db,
                                   enabled=guard_enabled)
        return True

    def set_bands(self, bands: list | None, loudness: bool = False) -> None:
        """
        Change the EQ curve mid-stream, keeping the filter state.

        The biquad state is carried across the coefficient change rather than
        zeroed: the section count is fixed by NUM_BANDS, so the state array
        still fits, and holding it means the filter continues from where the
        audio actually is. Zeroing would produce a transient at the moment of
        the change — precisely when someone is listening for the difference
        the change made.

        Going from flat to shaped allocates fresh (zero) state, which is
        correct: there was no filter running to carry.
        """
        if bands is None:
            bands = DEFAULT_BANDS
        if len(bands) != NUM_BANDS:
            bands = list(bands) + [0.0] * (NUM_BANDS - len(bands))

        if not loudness and all(b == 0.0 for b in bands):
            self._sos = None
            return

        sos = build_sos(bands, self._sample_rate, loudness)
        if self._sos is None or self._zi.shape[0] != sos.shape[0]:
            self._zi = np.zeros((sos.shape[0], 2), dtype=np.float64)
        self._sos = sos

    def set_volume_gain(self, gain: float) -> None:
        """Change the pre-chain volume gain; the next process() call ramps
        to it. Only meaningful on a chain built with volume_gain."""
        if self._vol_target is not None:
            self._vol_target = float(gain)

    def _apply_volume(self, x: np.ndarray) -> np.ndarray:
        # The same arithmetic as device/internal/outchain's pre-gain (and
        # speaker.softVolume before it): a linear ramp from the last gain to
        # the target across this call, accumulated one step at a time so the
        # rounding matches the Go loop's `g += step`.
        tgt, cur = self._vol_target, self._vol_cur
        if tgt == cur:
            out = x if tgt == 1.0 else x * tgt
        else:
            step = (tgt - cur) / x.size
            g = np.add.accumulate(np.concatenate(([cur + step],
                                                  np.full(x.size - 1, step))))
            out = x * g
        self._vol_cur = tgt
        return out

    def process(self, pcm: bytes) -> bytes:
        if len(pcm) < 2 or (self._sos is None and self._limiter is None
                            and self._guard is None and self._fir is None
                            and self._vol_target is None):
            return pcm
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
        if self._fir_bounds is not None:
            # From the target the volume is ramping to, read once per call,
            # as the device reads it once per period.
            self._fir.set_band(radar_eq_band(self._vol_target, self._fir_bounds))
        if self._vol_target is not None:
            samples = self._apply_volume(samples)
        if self._fir is not None:
            samples = self._fir.process(samples)
            samples, self._peq_zi = sosfilt(self._peq_sos, samples, zi=self._peq_zi)
        if self._sos is not None:
            samples, self._zi = sosfilt(self._sos, samples, zi=self._zi)
        if self._guard is not None:
            samples = self._guard.process(samples)
        if self._limiter is not None:
            samples = self._limiter.process(samples)
        if self._fir is not None:
            samples = samples * self._trim   # OutputTrim: after MBCL's limiter
        return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()

    def flush(self) -> bytes:
        """
        Emit the limiter's held look-ahead tail at end of stream.

        Returns empty when there is no limiter, so callers can call it
        unconditionally. Without it the last few ms of every music stream are
        dropped — inaudible on a track, obvious on a short announcement.
        """
        lat = getattr(self._guard, "latency", 0) if self._guard is not None else 0
        guard_tail = self._guard.process(np.zeros(lat)) if lat else None
        if self._limiter is None:
            if guard_tail is None:
                return b""
            tail = guard_tail
        else:
            tail = self._limiter.flush() if guard_tail is None else np.concatenate(
                (self._limiter.process(guard_tail), self._limiter.flush()))
        if not tail.size:
            return b""
        if self._fir is not None:
            tail = tail * self._trim
        return np.clip(tail, -32768, 32767).astype(np.int16).tobytes()


class Passthrough:
    """
    StreamingEQ's interface with nothing behind it, for a device that runs the
    output chain itself (the `output_chain` capability).

    The audio must leave here untouched, and every call site keeps its shape:
    `update` reports no change, so the music feed never logs a chain it is not
    running, and `flush` has no look-ahead tail to emit.
    """

    limiter = None
    guard = None

    def update(self, **_) -> bool:
        return False

    def process(self, pcm: bytes) -> bytes:
        return pcm

    def flush(self) -> bytes:
        return b""


# ─── Chain instrumentation ────────────────────────────────────────────────
#
# Whether the output chain is doing anything has been unanswerable from
# outside it, and that has now cost four listening tests. The stages
# interact hard enough that "I hear no difference" is NOT evidence either
# way: the bass guard is worth ~7.7dB of overall level at a modest EQ and
# ~0dB under a heavy boost, because the limiter gives back exactly what the
# guard takes away (measured 2026-08-20 — guard on/off at +12dB on all
# eight bands is -0.17dB overall, and the whole difference moves into the
# midrange instead). A listener judging by loudness is then judging the one
# cue that has been cancelled out.
#
# So the settings and the WORK DONE are reported separately. Settings say
# what reached the audio path, which is the config question; max reduction
# says whether the law ever engaged, which is the audio question. A guard
# that is enabled and reports 0.00dB of reduction is being fed content with
# no bass in it — a different fault from one that never got the setting.
#
# Cost is two log lines per stream plus one per live change, so nothing runs
# per frame. `max_reduction_db` is already maintained by both processors;
# this only surfaces it.


def _stage_state(stage, off="off") -> bool:
    """
    Whether a chain stage will actually process.

    Both shapes mean disabled and both occur: the one-shot path (em_eq.apply)
    takes None from for_stream(), while StreamingEQ always holds an instance
    and carries an `enabled` flag so a stream can be toggled without dropping
    filter state.
    """
    return stage is not None and getattr(stage, "enabled", True)


def describe_chain(bands, loudness, limiter=None, guard=None) -> str:
    """
    One line naming what this stream's chain is SET to.

    Emitted when a stream starts and again whenever a live update changes
    something, so the log shows both the starting point and the fact that a
    dashboard change reached the audio — which is the half that could not be
    seen before.
    """
    shaped = bands and any(float(b) != 0.0 for b in bands)
    eq = ("flat" if not shaped
          else "/".join(f"{float(b):+g}" if float(b) else "0" for b in bands))
    parts = [f"eq={eq}", f"speech_boost={'on' if loudness else 'off'}"]
    parts.append(
        f"guard={f'{guard.bass_guard_db:g}dB' if _stage_state(guard) else 'off'}"
    )
    parts.append(
        f"limiter={f'{limiter.threshold_db:g}dB/{limiter.release_ms:g}ms'}"
        if _stage_state(limiter) else "limiter=off"
    )
    return " ".join(parts)


def describe_activity(limiter=None, guard=None) -> str:
    """
    One line naming what the chain actually DID, in dB of gain reduction.

    `n/a` distinguishes a stage that was off from one that was on and never
    engaged — the two look identical from a listening seat and want opposite
    investigations.
    """
    def red(stage):
        if not _stage_state(stage):
            return "n/a"
        return f"{stage.max_reduction_db:.2f}dB"
    return f"guard_reduction={red(guard)} limiter_reduction={red(limiter)}"
