package outchain

import (
	_ "embed"
	"encoding/json"
	"math"
	"sync"
)

// Radar's stock FIR is five curves, chosen by volume: AFE.cfg's "Equalizer
// FIR" lists EQ_50/60/70/80/100.cfg against "Volume Boundary"
// [50,60,70,80,100]. They are a loudness compensation — EQ_50 boosts 80Hz
// by +10.1dB, EQ_100 by +1.4dB — not one curve at five gains, which is what
// biscuit's EQ files are. See controller/em_eq.py's _RADAR_EQ_BANDED_PATH,
// which this mirrors.
//
//go:embed radar_eq_banded.json
var radarEQBandedJSON []byte

type radarEQBandedFile struct {
	Boundaries []float64   `json:"boundaries"`
	Taps       [][]float64 `json:"taps"`
}

var (
	radarEQBandedOnce sync.Once
	radarEQBands      [][]float64
	radarEQBoundaries []float64
)

// loadRadarEQBands parses the embedded coefficients once. A corrupt embed,
// or one whose bands disagree in count or length, yields nil, which the
// chain treats as "no stock curve available" rather than crashing the
// device.
func loadRadarEQBands() ([][]float64, []float64) {
	radarEQBandedOnce.Do(func() {
		var f radarEQBandedFile
		if json.Unmarshal(radarEQBandedJSON, &f) != nil ||
			len(f.Taps) == 0 || len(f.Taps) != len(f.Boundaries) {
			return
		}
		for _, t := range f.Taps {
			if len(t) == 0 || len(t) != len(f.Taps[0]) {
				return
			}
		}
		radarEQBands, radarEQBoundaries = f.Taps, f.Boundaries
	})
	return radarEQBands, radarEQBoundaries
}

// speakerMusicCurve is Android's speaker music volume curve
// (audio_policy_volumes.xml, DEFAULT_DEVICE_CATEGORY_SPEAKER_VOLUME_CURVE):
// index 0-100 against attenuation in dB, linear in dB between points. The
// same curve turned stock's volume index into an attenuation, so it is the
// one that turns ours back into an index.
var speakerMusicCurve = [][2]float64{{1, -58}, {20, -40}, {60, -17}, {100, 0}}

// stockVolumeIndex is the 0-100 volume index stock would have been at to
// attenuate by gain — em_eq.stock_volume_index. Rounded half up to 1e-6, so
// an attenuation exactly on a boundary resolves the same way at both ends.
func stockVolumeIndex(gain float64) float64 {
	if gain <= 0 {
		return 0
	}
	att := 20 * math.Log10(gain)
	pts := speakerMusicCurve
	idx := pts[len(pts)-1][0]
	switch {
	case att >= pts[len(pts)-1][1]:
	case att <= pts[0][1]:
		idx = pts[0][0]
	default:
		for k := 0; k+1 < len(pts); k++ {
			i0, a0, i1, a1 := pts[k][0], pts[k][1], pts[k+1][0], pts[k+1][1]
			if att <= a1 {
				idx = i0 + (att-a0)/(a1-a0)*(i1-i0)
				break
			}
		}
	}
	return math.Floor(idx*1e6+0.5) / 1e6
}

// radarEQBand is which banded FIR stock plays at this volume gain: the
// first whose boundary is at or above the index — em_eq.radar_eq_band.
func radarEQBand(gain float64, boundaries []float64) int {
	idx := stockVolumeIndex(gain)
	for i, b := range boundaries {
		if idx <= b {
			return i
		}
	}
	return len(boundaries) - 1
}

// eqFIR runs Radar's stock speaker EQ — a long (2048-tap) FIR, not the
// biquad shape the rest of this package's eq type uses — via FFT-based
// overlap-save. See controller/em_eq.py's _OverlapSaveFIR, which this is
// held bit-exact to (testdata/gen_vectors.py + vectors.json).
//
// Fixed block size, unlike the Python reference: every production caller
// hands Process exactly one period (periodSamples) every time, so there
// is no equivalent of Python's arbitrary-chunk-length handling to carry —
// process panics if ever called with a different length, which is the
// correct failure mode for a bug in the caller rather than silently
// computing a wrong answer. Tests exercise exactly periodSamples per
// call, same as production, and prove the general overlap-save MATH via
// fft_test.go instead (which is not tied to any one block size).
type eqFIR struct {
	h        []complex128   // the current band's taps, zero-padded and transformed
	hs       [][]complex128 // every band's, the same way; h is one of these
	band     int
	fadeFrom int          // band to crossfade FROM on the next period, or -1
	fadeBuf  []complex128 // the old band's product during a crossfade
	m        int          // filter length (taps)
	period   int          // fixed new-samples-per-call this was sized for
	fftSize  int
	overlap  []float64 // last m-1 raw input samples carried across periods

	// Reused across every call — one FFT-sized heap buffer and one output
	// buffer, allocated once here rather than per period. This runs every
	// ~42.7ms on hardware the rest of this package is visibly tuned to
	// keep allocation-free (device/CLAUDE.md's measured per-period costs);
	// a fresh 4096-complex128 buffer every period is exactly the kind of
	// GC pressure that tuning exists to avoid.
	buf []complex128
	out []float64
}

// newEQFIR builds an overlap-save FIR for taps, sized for exactly
// periodSamples new samples per call. Returns nil if taps is empty (no
// data loaded) — callers must treat a nil *eqFIR as "run nothing",
// mirroring the controller's fallback when radar_eq_taps.json is absent.
func newEQFIR(taps []float64, periodSamples int) *eqFIR {
	if len(taps) == 0 {
		return nil
	}
	return newEQFIRBands([][]float64{taps}, periodSamples)
}

// newEQFIRBands is newEQFIR over several filters of equal length, starting
// on band 0, switched with setBand. Nil for no bands or unequal lengths.
func newEQFIRBands(bands [][]float64, periodSamples int) *eqFIR {
	if len(bands) == 0 || len(bands[0]) == 0 {
		return nil
	}
	m := len(bands[0])
	fftSize := nextPow2(m - 1 + periodSamples)

	hs := make([][]complex128, len(bands))
	for b, taps := range bands {
		if len(taps) != m {
			return nil
		}
		h := make([]complex128, fftSize)
		for i, v := range taps {
			h[i] = complex(v, 0)
		}
		fft(h, false)
		hs[b] = h
	}

	return &eqFIR{
		h:        hs[0],
		hs:       hs,
		fadeFrom: -1,
		m:        m,
		period:   periodSamples,
		fftSize:  fftSize,
		overlap:  make([]float64, m-1),
		buf:      make([]complex128, fftSize),
		out:      make([]float64, periodSamples),
	}
}

// process filters exactly f.period samples, returning f.out — reused
// across calls, so the result must be consumed before the next call to
// process (true of every call site in this package: Chain.Process reads
// it back within the same period it was produced in, never retains it
// across periods). Panics if len(x) != f.period — see the type doc.
func (f *eqFIR) process(x []float64) []float64 {
	if len(x) != f.period {
		panic("eqFIR.process: called with a different length than newEQFIR was sized for")
	}

	buf := f.buf
	for i := range buf {
		buf[i] = 0
	}
	for i, v := range f.overlap {
		buf[i] = complex(v, 0)
	}
	for i, v := range x {
		buf[f.m-1+i] = complex(v, 0)
	}

	fft(buf, false)
	fading := f.fadeFrom >= 0
	if fading {
		if f.fadeBuf == nil {
			f.fadeBuf = make([]complex128, f.fftSize)
		}
		old := f.hs[f.fadeFrom]
		for i := range buf {
			f.fadeBuf[i] = buf[i] * old[i]
		}
		fft(f.fadeBuf, true)
		f.fadeFrom = -1
	}
	for i := range buf {
		buf[i] *= f.h[i]
	}
	fft(buf, true)

	out := f.out
	for i := range out {
		out[i] = real(buf[f.m-1+i])
	}
	if fading {
		// Linear crossfade across the period from the old band's output to
		// the new one's — both filtered the same input history, so this is
		// a change of curve, faded so it does not land as a step.
		n := float64(len(out))
		for i := range out {
			w := float64(i+1) / n
			out[i] = real(f.fadeBuf[f.m-1+i])*(1-w) + out[i]*w
		}
	}

	// Carry the last m-1 RAW input samples for the next call — history of
	// the signal, not of the filtered output. period is fixed and the
	// overlap is always exactly m-1 long, so (since period > 0) the new
	// overlap is always drawn entirely from x: the old overlap has fully
	// rolled off UNLESS period < m-1, in which case part of it survives.
	// Updated IN PLACE (no new allocation) — copy() is memmove-like, so
	// the keep-branch's overlapping src/dst within f.overlap is safe.
	if f.period >= f.m-1 {
		copy(f.overlap, x[f.period-(f.m-1):])
	} else {
		keep := f.m - 1 - f.period
		copy(f.overlap[:keep], f.overlap[f.period:f.period+keep])
		copy(f.overlap[keep:], x)
	}

	return out
}

// setBand switches filter; the next process crossfades into it. Setting the
// band it is already on does nothing.
func (f *eqFIR) setBand(b int) {
	if b == f.band || b < 0 || b >= len(f.hs) {
		return
	}
	f.fadeFrom = f.band
	f.band = b
	f.h = f.hs[b]
}

// startOn puts the filter on band b with no crossfade — for its first
// period, where there is no previous curve to fade from.
func (f *eqFIR) startOn(b int) {
	if b < 0 || b >= len(f.hs) {
		return
	}
	f.band, f.h, f.fadeFrom = b, f.hs[b], -1
}

func (f *eqFIR) reset() {
	for i := range f.overlap {
		f.overlap[i] = 0
	}
}
