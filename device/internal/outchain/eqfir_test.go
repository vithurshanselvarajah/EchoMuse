package outchain

import (
	"math"
	"math/rand"
	"testing"
)

// naiveFilter is the causal FIR filtering of x by h — y[n] = sum_k h[k]*x[n-k]
// with x treated as zero before index 0 — computed with no FFT at all, as
// an independent ground truth.
func naiveFilter(x, h []float64) []float64 {
	y := make([]float64, len(x))
	for n := range x {
		var s float64
		for k := 0; k < len(h) && k <= n; k++ {
			s += h[k] * x[n-k]
		}
		y[n] = s
	}
	return y
}

func randomSlice(rng *rand.Rand, n int, scale float64) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = rng.NormFloat64() * scale
	}
	return s
}

// The core correctness test: streaming eqFIR over many fixed-size periods
// must match ONE naive convolution of the whole concatenated signal — this
// is the test that would catch an overlap bookkeeping bug, since a wrong
// carry only shows up across a call boundary.
func TestEQFIRMatchesNaiveFilterAcrossManyPeriods(t *testing.T) {
	cases := []struct {
		name       string
		filterLen  int
		period     int
		numPeriods int
	}{
		{"filter_smaller_than_period", 17, 64, 20},
		{"filter_equal_to_period", 64, 64, 20},
		{"filter_larger_than_period", 2048, 256, 10},
		{"radar_shape", 2048, 2048, 8}, // the actual production sizing
		{"single_tap_period_one", 1, 1, 50},
		{"filter_len_two", 2, 1, 30},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(42))
			h := randomSlice(rng, c.filterLen, 0.1)
			x := randomSlice(rng, c.period*c.numPeriods, 1.0)

			want := naiveFilter(x, h)

			fir := newEQFIR(h, c.period)
			got := make([]float64, 0, len(x))
			for i := 0; i < c.numPeriods; i++ {
				chunk := x[i*c.period : (i+1)*c.period]
				got = append(got, fir.process(chunk)...)
			}

			for i := range want {
				if math.Abs(got[i]-want[i]) > 1e-8 {
					t.Fatalf("%s: sample %d (period %d): got %v want %v",
						c.name, i, i/c.period, got[i], want[i])
				}
			}
		})
	}
}

func TestEQFIRReturnsNilForEmptyTaps(t *testing.T) {
	if newEQFIR(nil, 2048) != nil {
		t.Fatal("newEQFIR(nil, ...) must return nil, not a FIR that would panic on use")
	}
	if newEQFIR([]float64{}, 2048) != nil {
		t.Fatal("newEQFIR([]float64{}, ...) must return nil")
	}
}

func TestEQFIRPanicsOnWrongLength(t *testing.T) {
	fir := newEQFIR([]float64{1, 0, 0}, 8)
	defer func() {
		if recover() == nil {
			t.Fatal("expected a panic calling process with the wrong length")
		}
	}()
	fir.process(make([]float64, 7))
}

func TestEQFIRResetClearsHistory(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	h := randomSlice(rng, 64, 0.1)

	a := newEQFIR(h, 32)
	a.process(randomSlice(rng, 32, 1.0)) // prime history

	b := newEQFIR(h, 32)
	a.reset()

	y := randomSlice(rng, 32, 1.0)
	gotA := a.process(y)
	gotB := b.process(y)
	for i := range gotA {
		if math.Abs(gotA[i]-gotB[i]) > 1e-10 {
			t.Fatalf("index %d: after reset=%v fresh=%v", i, gotA[i], gotB[i])
		}
	}
}

func TestEQFIRZeroFilterIsSilence(t *testing.T) {
	fir := newEQFIR(make([]float64, 2048), 2048)
	x := make([]float64, 2048)
	for i := range x {
		x[i] = 12345.0
	}
	out := fir.process(x)
	for i, v := range out {
		if v != 0 {
			t.Fatalf("sample %d: got %v, want 0", i, v)
		}
	}
}

func TestEQFIRImpulseFilterIsIdentity(t *testing.T) {
	h := make([]float64, 2048)
	h[0] = 1.0
	fir := newEQFIR(h, 512)
	rng := rand.New(rand.NewSource(9))
	x := randomSlice(rng, 512, 1.0)
	out := fir.process(x)
	for i := range x {
		if math.Abs(out[i]-x[i]) > 1e-10 {
			t.Fatalf("sample %d: got %v want %v (impulse filter must be identity)", i, out[i], x[i])
		}
	}
}

// Confirms the embedded data actually loads and is the right shape — a
// corrupt embed would otherwise only surface as the stock curve silently
// not running.
func TestLoadRadarEQBandsFromEmbed(t *testing.T) {
	bands, bounds := loadRadarEQBands()
	if len(bands) != 5 || len(bounds) != 5 {
		t.Fatalf("got %d bands / %d boundaries, want 5/5", len(bands), len(bounds))
	}
	want := []float64{50, 60, 70, 80, 100}
	for i, b := range bands {
		if len(b) != 2048 {
			t.Errorf("band %d: %d taps, want 2048", i, len(b))
		}
		if bounds[i] != want[i] {
			t.Errorf("boundary %d = %g, want %g", i, bounds[i], want[i])
		}
	}
}

// The volume index is recovered through Android's speaker curve, and a
// level landing exactly on a boundary must resolve to that boundary's own
// band — the same answers em_eq.radar_eq_band gives, pinned on both sides.
func TestRadarEQBandByVolume(t *testing.T) {
	bounds := []float64{50, 60, 70, 80, 100}
	gain := func(level int) float64 {
		if level == 0 {
			return 0
		}
		return math.Pow(10, float64(level-127)/40)
	}
	for _, tc := range []struct{ level, band int }{
		{0, 0}, {47, 0}, {81, 0}, {82, 1}, {93, 1}, {94, 2},
		{101, 2}, {102, 3}, {110, 3}, {111, 4}, {127, 4},
	} {
		if got := radarEQBand(gain(tc.level), bounds); got != tc.band {
			t.Errorf("level %d: band %d, want %d (index %g)",
				tc.level, got, tc.band, stockVolumeIndex(gain(tc.level)))
		}
	}
}

// A band switch crossfades: the period it happens in starts on the old
// curve's output and ends on the new one's, and the period after is the
// new curve alone — the same as a filter that was on it all along.
func TestEQFIRBandSwitchCrossfades(t *testing.T) {
	const n = 256
	a := make([]float64, 32)
	b := make([]float64, 32)
	a[0], b[0] = 1, 0.25 // two pure gains, so the outputs are easy to state
	f := newEQFIRBands([][]float64{a, b}, n)
	x := make([]float64, n)
	for i := range x {
		x[i] = 1000
	}
	f.process(x)
	f.setBand(1)
	y := append([]float64(nil), f.process(x)...)
	if math.Abs(y[0]-(1000*(1-1.0/n)+250/float64(n))) > 1e-6 {
		t.Errorf("first sample of the fade = %g", y[0])
	}
	if math.Abs(y[n-1]-250) > 1e-6 {
		t.Errorf("last sample of the fade = %g, want 250", y[n-1])
	}
	y = f.process(x)
	for i, v := range y {
		if math.Abs(v-250) > 1e-6 {
			t.Fatalf("after the fade, sample %d = %g, want 250", i, v)
		}
	}
}
