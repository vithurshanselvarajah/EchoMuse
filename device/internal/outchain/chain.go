package outchain

import (
	"fmt"
	"math"
	"strings"
	"sync"
	"sync/atomic"
)

// Params is the chain's whole configuration. Defaults match
// em_db.DEFAULT_DEVICE_CONFIG.
type Params struct {
	Bands              [NumBands]float64
	Loudness           bool
	GuardEnabled       bool
	GuardDb            float64
	LimiterEnabled     bool
	LimiterThresholdDb float64
	LimiterReleaseMs   float64
	// StockCurve runs Radar's own stock FIR EQ (eqFIR) ahead of the 8 bands
	// above, additively — see controller/em_eq.py's stock_curve. Has no
	// effect at all on a board without a loaded curve (only Radar, and
	// only when the embedded taps parsed), same as the controller leaving
	// it unused for every other board.
	StockCurve bool
}

// DefaultParams mirrors the controller's defaults, so a device that has not
// yet had a config push sounds the way the controller would have made it.
func DefaultParams() Params {
	return Params{
		GuardEnabled:       true,
		GuardDb:            -30,
		LimiterEnabled:     true,
		LimiterThresholdDb: -1,
		LimiterReleaseMs:   150,
	}
}

// String is the one-line description em_eq.describe_chain gives, so device
// and controller logs read the same way.
func (p Params) String() string {
	eqs := "flat"
	if !isFlat(p.Bands, false) {
		parts := make([]string, NumBands)
		for i, b := range p.Bands {
			if b == 0 {
				parts[i] = "0"
			} else {
				parts[i] = fmt.Sprintf("%+g", b)
			}
		}
		eqs = strings.Join(parts, "/")
	}
	boost := "off"
	if p.Loudness {
		boost = "on"
	}
	guard := "off"
	if p.GuardEnabled {
		guard = fmt.Sprintf("%gdB", math.Min(p.GuardDb, 0))
	}
	lim := "off"
	if p.LimiterEnabled {
		lim = fmt.Sprintf("%gdB/%gms", math.Min(p.LimiterThresholdDb, 0), p.LimiterReleaseMs)
	}
	return fmt.Sprintf("eq=%s speech_boost=%s guard=%s limiter=%s", eqs, boost, guard, lim)
}

// Chain runs EQ → bass guard → limiter on stereo S16_LE periods.
//
// Order is em_eq's: the guard removes excursion the driver cannot deliver,
// THEN the limiter catches what is left. Limiting first would spend gain
// reduction on bass about to be thrown away.
//
// The processing is MONO: L and R are averaged, processed once and written
// back to both. The wire is mono and toStereo duplicates it, so the average
// is exact on everything this device plays; stereo output is not supported
// on this hardware (device/CLAUDE.md), and processing two identical channels
// would double the cost for nothing.
//
// Process runs on the ALSA write goroutine only. SetParams and SetActive may
// be called from anywhere; they take effect at the next period.
type Chain struct {
	fs float64

	active atomic.Bool // false: Process is a passthrough

	mu      sync.Mutex
	pending *Params // set by SetParams, taken by Process

	// Owned by the ALSA goroutine.
	params  Params
	eq      eq
	guard   *bassGuard
	lim     *limiter
	idle    bool // state is all zero and input is silence
	running bool // active on the previous period

	// Stock FIR curve (Radar only — nil taps on every other board, and
	// newEQFIR(nil, ...) is nil, so fir stays nil there with no extra
	// gating needed at this level).
	firTaps  []float64 // resolved once at construction; nil = unavailable
	fir      *eqFIR    // lazily sized to the first period's length
	wantFIR  bool      // params.StockCurve as of the last apply()
	firScratch []float64 // reused per period — no per-call allocation
}

// New builds a chain at the given sample rate, inactive, with DefaultParams,
// tuned for biscuit — see NewForBoard for a board-aware chain. Kept so every
// existing caller and test vector (biscuit-only, to date) is unaffected.
func New(sampleRate int) *Chain {
	return NewForBoard(sampleRate, "biscuit")
}

// NewForBoard builds a chain at the given sample rate, inactive, with
// DefaultParams, with the bass guard tuned for boardID (pkg/board.IDOf) —
// see bassGuardTuning. The guard always varies by board; the stock FIR
// curve is only ever available on "radar" (nil firTaps on every other
// board id, so StockCurve has no effect there regardless of config). The
// EQ bands and limiter are not board-specific.
func NewForBoard(sampleRate int, boardID string) *Chain {
	fs := float64(sampleRate)
	c := &Chain{
		fs:    fs,
		eq:    eq{fs: fs},
		guard: newBassGuard(fs, boardID),
		lim:   newLimiter(fs),
		idle:  true,
	}
	if boardID == "radar" {
		c.firTaps = loadRadarEQTaps()
	}
	c.apply(DefaultParams())
	return c
}

// SetActive turns processing on or off. Off is a passthrough, which is what
// a device must do while its controller is still processing the audio itself:
// the chain applied twice is a doubled EQ curve and a second limiter.
func (c *Chain) SetActive(on bool) { c.active.Store(on) }

// Active reports whether the chain is processing.
func (c *Chain) Active() bool { return c.active.Load() }

// SetParams queues a new configuration for the next period. Everything is
// updated in place — filter states, the limiter's delay line and its gain
// carry across — so a change mid-song does not click.
func (c *Chain) SetParams(p Params) {
	c.mu.Lock()
	c.pending = &p
	c.mu.Unlock()
}

func (c *Chain) apply(p Params) {
	c.params = p
	c.eq.set(p.Bands, p.Loudness)
	c.guard.enabled = p.GuardEnabled
	c.guard.floorDb = math.Min(p.GuardDb, 0)
	c.lim.enabled = p.LimiterEnabled
	c.lim.setParams(p.LimiterThresholdDb, p.LimiterReleaseMs, c.fs)
	// Actually turning the FIR on/off is deferred to Process, which is the
	// only place that knows this period's frame count (needed to size it)
	// — apply only records what is WANTED. No effect at all when firTaps
	// is nil (every board but Radar).
	c.wantFIR = p.StockCurve && c.firTaps != nil
}

// takePending applies a queued SetParams. Returns the params that are now in
// force and whether they changed.
func (c *Chain) takePending() (Params, bool) {
	c.mu.Lock()
	p := c.pending
	c.pending = nil
	c.mu.Unlock()
	if p == nil {
		return c.params, false
	}
	changed := *p != c.params
	c.apply(*p)
	return c.params, changed
}

// Idle reports whether processing a silent period would return silence
// unchanged, so the caller can skip it. True while inactive.
func (c *Chain) Idle() bool {
	return !c.active.Load() || c.idle
}

// Process runs the chain over one stereo S16_LE period, IN PLACE, and returns
// the same buffer. The caller must own buf — never pass a shared silence
// buffer.
//
// The first return is non-nil only when a queued SetParams changed the chain
// on this period, so the caller can log what the audio is now going through.
func (c *Chain) Process(buf []byte) (applied *Params) {
	if p, changed := c.takePending(); changed {
		applied = &p
	}
	active := c.active.Load()
	if active != c.running {
		// Entering or leaving: the chain's state belongs to audio it last
		// saw, which is not the audio arriving now. Start clean.
		c.reset()
		c.running = active
	}
	if !active {
		return applied
	}

	frames := len(buf) / 4

	// FIR on/off is decided in apply(), but SIZED here — this is the first
	// point the chain knows the period's frame count. In production this
	// never changes between calls, so sizing happens once; a device that
	// somehow called Process with a varying frames count would panic
	// inside eqFIR.process, which is the right failure for that bug
	// rather than a wrong answer.
	if c.wantFIR && c.fir == nil {
		c.fir = newEQFIR(c.firTaps, frames)
		if c.firScratch == nil || len(c.firScratch) != frames {
			c.firScratch = make([]float64, frames)
		}
	} else if !c.wantFIR && c.fir != nil {
		c.fir = nil
	}
	if c.fir != nil {
		for i := 0; i < frames; i++ {
			off := i * 4
			l := int16(uint16(buf[off]) | uint16(buf[off+1])<<8)
			r := int16(uint16(buf[off+2]) | uint16(buf[off+3])<<8)
			c.firScratch[i] = (float64(l) + float64(r)) / 2
		}
		copy(c.firScratch, c.fir.process(c.firScratch))
	}

	silentIn, silentOut := true, true
	for i := 0; i < frames; i++ {
		off := i * 4
		l := int16(uint16(buf[off]) | uint16(buf[off+1])<<8)
		r := int16(uint16(buf[off+2]) | uint16(buf[off+3])<<8)
		if l != 0 || r != 0 {
			silentIn = false
		}
		var x float64
		if c.fir != nil {
			// Radar's stock curve, layered ahead of the bands below —
			// additive with them, same as controller/em_eq.py's
			// StreamingEQ(stock_curve=True), never a replacement.
			x = c.firScratch[i]
		} else {
			x = (float64(l) + float64(r)) / 2
		}

		x = c.eq.step(x)
		x = c.guard.step(x)
		x = c.lim.step(x)

		// Backstop, then truncation toward zero — np.clip(...).astype(int16)
		// in the reference.
		if x > ceiling {
			x = ceiling
		} else if x < -fullScale {
			x = -fullScale
		}
		s := int16(x)
		if s != 0 {
			silentOut = false
		}
		lo, hi := byte(uint16(s)), byte(uint16(s)>>8)
		buf[off], buf[off+1], buf[off+2], buf[off+3] = lo, hi, lo, hi
	}

	// A silent period that came out silent means every filter tail has
	// decayed below one LSB. Zero the state and stop processing silence
	// until audio returns — otherwise the chain runs flat out on an idle
	// speaker, forever. The reset moves the output by less than one LSB.
	if silentIn && silentOut {
		c.reset()
		c.idle = true
	} else {
		c.idle = false
	}
	return applied
}

func (c *Chain) reset() {
	c.eq.reset()
	c.guard.reset()
	c.lim.reset()
	if c.fir != nil {
		c.fir.reset()
	}
	c.idle = true
}

// Stats is the chain's instrumentation: the WORK done, as against Params,
// which is what it was set to. A stage that is on and reports 0.00dB never
// engaged, which is a different fault from one that is off.
type Stats struct {
	GuardReductionDb   float64
	LimiterReductionDb float64
	Clipped            uint64 // must stay 0 while limiting
	ClippedBypassed    uint64
}

// TakeStats returns and clears the maximum reductions since the last call.
// ALSA goroutine only.
func (c *Chain) TakeStats() Stats {
	s := Stats{
		GuardReductionDb:   c.guard.maxReductionDb,
		LimiterReductionDb: c.lim.maxReductionDb,
		Clipped:            c.lim.clipped,
		ClippedBypassed:    c.lim.clippedBypassed,
	}
	c.guard.maxReductionDb, c.lim.maxReductionDb = 0, 0
	return s
}
