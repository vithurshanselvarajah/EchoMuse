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
		name          string
		filterLen     int
		period        int
		numPeriods    int
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

// Confirms the embedded data actually loads and is the right size — a
// corrupt embed would otherwise only surface as a panic deep inside
// newEQFIR in production.
func TestLoadRadarEQTapsFromEmbed(t *testing.T) {
	taps := loadRadarEQTaps()
	if len(taps) != 2048 {
		t.Fatalf("got %d taps, want 2048", len(taps))
	}
}
