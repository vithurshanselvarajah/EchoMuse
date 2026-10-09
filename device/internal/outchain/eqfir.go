package outchain

import (
	_ "embed"
	"encoding/json"
	"sync"
)

//go:embed radar_eq_taps.json
var radarEQTapsJSON []byte

type radarEQTapsFile struct {
	Taps []float64 `json:"taps"`
}

var (
	radarEQTapsOnce sync.Once
	radarEQTaps     []float64
)

// loadRadarEQTaps parses the embedded coefficients once. A corrupt or
// missing embed (impossible via go:embed at compile time, but checked
// anyway since a malformed JSON would otherwise panic deep inside eqFIR's
// constructor) yields a nil slice, which newEQFIR treats as "no stock
// curve available" rather than crashing the device.
func loadRadarEQTaps() []float64 {
	radarEQTapsOnce.Do(func() {
		var f radarEQTapsFile
		if json.Unmarshal(radarEQTapsJSON, &f) == nil && len(f.Taps) > 0 {
			radarEQTaps = f.Taps
		}
	})
	return radarEQTaps
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
	h       []complex128 // taps, zero-padded to fftSize, pre-transformed
	m       int          // filter length (taps)
	period  int          // fixed new-samples-per-call this was sized for
	fftSize int
	overlap []float64 // last m-1 raw input samples carried across periods

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
	m := len(taps)
	if m == 0 {
		return nil
	}
	fftSize := nextPow2(m - 1 + periodSamples)

	h := make([]complex128, fftSize)
	for i, v := range taps {
		h[i] = complex(v, 0)
	}
	fft(h, false)

	return &eqFIR{
		h:       h,
		m:       m,
		period:  periodSamples,
		fftSize: fftSize,
		overlap: make([]float64, m-1),
		buf:     make([]complex128, fftSize),
		out:     make([]float64, periodSamples),
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
	for i := range buf {
		buf[i] *= f.h[i]
	}
	fft(buf, true)

	out := f.out
	for i := range out {
		out[i] = real(buf[f.m-1+i])
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

func (f *eqFIR) reset() {
	for i := range f.overlap {
		f.overlap[i] = 0
	}
}
