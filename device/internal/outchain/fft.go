package outchain

import "math"

// fft is an in-place iterative radix-2 Cooley-Tukey FFT (decimation in
// time). len(a) must be a power of 2 — the one caller (eqFIR) only ever
// calls it at the fixed size it precomputed, so that is asserted once
// there rather than checked on every call.
//
// There is no FFT in the Go standard library and this project deliberately
// carries no DSP dependency (device/CLAUDE.md: new deps must cross-compile
// cleanly against the FireOS 5 sysroot, and a pure-Go implementation this
// size is lower risk than vetting a third party's). Correctness here rests
// entirely on fft_test.go, not on this comment — in particular on
// comparing convolution results against an INDEPENDENT naive convolution,
// not merely a round-trip of this same function.
func fft(a []complex128, inverse bool) {
	n := len(a)
	if n <= 1 {
		return
	}

	// Bit-reversal permutation.
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}

	for length := 2; length <= n; length <<= 1 {
		angle := -2 * math.Pi / float64(length)
		if inverse {
			angle = -angle
		}
		wlen := complex(math.Cos(angle), math.Sin(angle))
		half := length / 2
		for i := 0; i < n; i += length {
			w := complex(1.0, 0.0)
			for j := 0; j < half; j++ {
				u := a[i+j]
				v := a[i+j+half] * w
				a[i+j] = u + v
				a[i+j+half] = u - v
				w *= wlen
			}
		}
	}

	if inverse {
		scale := complex(float64(n), 0)
		for i := range a {
			a[i] /= scale
		}
	}
}

// isPow2 reports whether n is a positive power of 2.
func isPow2(n int) bool {
	return n > 0 && n&(n-1) == 0
}

// nextPow2 is the smallest power of 2 >= n.
func nextPow2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}
