package outchain

import "math"

// Radar's full 4-band MBCL ("Radar Tuning V4.5"), read verbatim off a Radar
// unit's own /system/vendor/etc/audio-algorithms/MBCL.cfg. This is the
// bit-exact Go mirror of controller/em_mbc.py's RadarMultiband — see that
// class's docstring for the full derivation: the real config table, why a
// naive recursive crossover split does NOT sum flat (1.59dB of measured
// ripple) and the allpass-compensation fix that makes it exact (4.6e-11dB),
// and the one number that is inferred rather than read (each band's
// compressor reuses its own limiter's release — the config has no
// comp_release field at all).
const (
	radarFc1 = 70.0
	radarFc2 = 200.0
	radarFc3 = 3250.0

	// The system gain MBCL applies to the WHOLE signal before splitting
	// into bands — not a per-band trim. Applied unconditionally, same as
	// every band's own comp_inVol/lim_inVol below — see radarMultiband.step.
	radarMbclInVolDb = 4.0
)

// radarBandSpec is one row of MBCL.cfg's "Bands Definition".
type radarBandSpec struct {
	compRatio, compThresholdDb, compFloorDb, compInVolDb float64
	limThresholdDb, limReleaseMs, limInVolDb             float64
}

var radarBands = [4]radarBandSpec{
	// Band 1: 0-70Hz. comp_ratio/comp_threshold agree exactly with
	// bassRatio/radarBassThresholdDb above — the same measured band read
	// twice. Its own limiter (lim_thresh -12dB, release 200ms) is new:
	// the single-band bassGuard this class replaces for Radar never had
	// one.
	{compRatio: 20.0, compThresholdDb: -25.0, compFloorDb: -40.0, compInVolDb: 0.0,
		limThresholdDb: -12.0, limReleaseMs: 200.0, limInVolDb: 0.0},
	// Band 2: 70-200Hz.
	{compRatio: 10.0, compThresholdDb: -18.0, compFloorDb: -40.0, compInVolDb: 0.0,
		limThresholdDb: -12.0, limReleaseMs: 80.0, limInVolDb: 0.0},
	// Band 3: 200-3250Hz — most of the midrange, and the only band with a
	// +3dB input trim into BOTH its compressor and its limiter.
	{compRatio: 3.0, compThresholdDb: -15.0, compFloorDb: -40.0, compInVolDb: 3.0,
		limThresholdDb: -4.0, limReleaseMs: 20.0, limInVolDb: 3.0},
	// Band 4: 3250Hz-Nyquist.
	{compRatio: 2.0, compThresholdDb: -10.0, compFloorDb: -40.0, compInVolDb: 3.0,
		limThresholdDb: -3.0, limReleaseMs: 20.0, limInVolDb: 0.0},
}

// Precomputed once: the fixed trims never change mid-stream, so there is no
// reason to pay an Exp call per sample per band for them.
var (
	radarSysGain       = dbToGain(radarMbclInVolDb)
	radarCompInVolGain [4]float64
	radarLimInVolGain  [4]float64
)

func init() {
	for i, b := range radarBands {
		radarCompInVolGain[i] = dbToGain(b.compInVolDb)
		radarLimInVolGain[i] = dbToGain(b.limInVolDb)
	}
}

// bandGain is one dynamics stage's detector and gain computer — em_mbc.py's
// _BandGain as a per-sample recursion. A compressor with ratio and floorDb;
// an infinite ratio and a -infinite floor (newBandLimiter) make it a pure
// peak limiter instead, reusing the exact same law the compressor uses —
// 1/+Inf is well-defined as 0 in IEEE754, so no branch is needed to tell
// the two apart.
type bandGain struct {
	ratio, thresholdDb, thresholdLin, floorDb float64
	slew           float64
	gainDb         float64
	gain           gainCache
	maxReductionDb float64
}

func newBandGain(ratio, thresholdDb, releaseMs, floorDb, fs float64) *bandGain {
	return &bandGain{
		ratio:        math.Max(1.0, ratio),
		thresholdDb:  thresholdDb,
		thresholdLin: fullScale * math.Pow(10, thresholdDb/20),
		floorDb:      math.Min(0.0, floorDb),
		slew:         releaseReferenceDb / (math.Max(0.1, releaseMs) / 1000) / fs,
	}
}

func newBandLimiter(thresholdDb, releaseMs, fs float64) *bandGain {
	return newBandGain(math.Inf(1), thresholdDb, releaseMs, math.Inf(-1), fs)
}

// levelGain returns the linear gain for this sample given x as the
// detector input, and advances the carried envelope — same optimisation as
// bassGuard.step: the log is skipped entirely below the threshold, compared
// in LINEAR units so the skip can never disagree with the dB comparison it
// stands in for.
func (b *bandGain) levelGain(x float64) float64 {
	target := 0.0
	if a := math.Abs(x); a > b.thresholdLin {
		levelDb := 20 * math.Log10(a/fullScale)
		target = max(-(levelDb-b.thresholdDb)*(1-1/b.ratio), b.floorDb)
	}
	b.gainDb = min(target, b.gainDb+b.slew)
	if r := -b.gainDb; r > b.maxReductionDb {
		b.maxReductionDb = r
	}
	return b.gain.of(b.gainDb)
}

func (b *bandGain) reset() { b.gainDb = 0 }

// radarMultiband is Radar's full MBCL — three crossovers splitting the
// signal into four bands, each running its own compressor then its own
// limiter, both with their own input trim; the four are summed. The
// combined full-band limiter (radarLimiterThresholdDb/ReleaseMs in
// limiter.go) is NOT part of this type — it is MBCL's own "Full-band
// limiter" entry and stays exactly where it already runs, downstream of
// this, in chain.go.
//
// EXACT FLATNESS, THROUGH ALLPASS COMPENSATION — see RadarMultiband's
// Python docstring for the derivation. Band 1 (which only ever sees the
// fc1 split) is additionally run through an allpass of fc2 and then of
// fc3; band 2 (fc1 and fc2) is additionally run through an allpass of
// fc3; bands 3 and 4 already carry all three splits' worth of filtering.
// Every *c/*c1/*c2 filter below is an INDEPENDENT instance of the same
// coefficients as the real split it stands in for — same transfer
// function, separate state, because it filters a different signal.
type radarMultiband struct {
	enabled bool

	lp1, hp1     [2]biquad // fc1 split, on x
	lp2, hp2     [2]biquad // fc2 split, on high1
	lp2c, hp2c   [2]biquad // fc2 compensation, on low1
	lp3, hp3     [2]biquad // fc3 split, on high2 -> band3/4
	lp3c1, hp3c1 [2]biquad // fc3 compensation, on low2 -> band2
	lp3c2, hp3c2 [2]biquad // fc3 compensation, on ap2(low1) -> band1

	comp [4]*bandGain
	lim  [4]*bandGain
}

func newRadarMultiband(fs float64) *radarMultiband {
	lo1, hi1 := butter2(radarFc1, fs, false), butter2(radarFc1, fs, true)
	lo2, hi2 := butter2(radarFc2, fs, false), butter2(radarFc2, fs, true)
	lo3, hi3 := butter2(radarFc3, fs, false), butter2(radarFc3, fs, true)

	m := &radarMultiband{
		lp1: [2]biquad{lo1, lo1}, hp1: [2]biquad{hi1, hi1},
		lp2: [2]biquad{lo2, lo2}, hp2: [2]biquad{hi2, hi2},
		lp2c: [2]biquad{lo2, lo2}, hp2c: [2]biquad{hi2, hi2},
		lp3: [2]biquad{lo3, lo3}, hp3: [2]biquad{hi3, hi3},
		lp3c1: [2]biquad{lo3, lo3}, hp3c1: [2]biquad{hi3, hi3},
		lp3c2: [2]biquad{lo3, lo3}, hp3c2: [2]biquad{hi3, hi3},
	}
	for i, b := range radarBands {
		m.comp[i] = newBandGain(b.compRatio, b.compThresholdDb, b.limReleaseMs, b.compFloorDb, fs)
		m.lim[i] = newBandLimiter(b.limThresholdDb, b.limReleaseMs, fs)
	}
	return m
}

func (m *radarMultiband) step(x float64) float64 {
	low1 := m.lp1[1].step(m.lp1[0].step(x))
	high1 := m.hp1[1].step(m.hp1[0].step(x))

	low2 := m.lp2[1].step(m.lp2[0].step(high1))
	high2 := m.hp2[1].step(m.hp2[0].step(high1))

	ap2Low1 := m.lp2c[1].step(m.lp2c[0].step(low1)) + m.hp2c[1].step(m.hp2c[0].step(low1))

	band3Raw := m.lp3[1].step(m.lp3[0].step(high2))
	band4Raw := m.hp3[1].step(m.hp3[0].step(high2))

	band2Raw := m.lp3c1[1].step(m.lp3c1[0].step(low2)) + m.hp3c1[1].step(m.hp3c1[0].step(low2))
	band1Raw := m.lp3c2[1].step(m.lp3c2[0].step(ap2Low1)) + m.hp3c2[1].step(m.hp3c2[0].step(ap2Low1))

	raw := [4]float64{band1Raw, band2Raw, band3Raw, band4Raw}

	// mbcl_inVol and every band's comp_inVol/lim_inVol apply EITHER WAY —
	// fixed gain stages, not dynamics ones. Gating them on `enabled` would
	// make the toggle step the level by as much as 6dB (band 3's
	// comp_inVol+lim_inVol) on top of whatever the law itself was doing.
	// Only levelGain() — the compression/limiting itself — is skipped
	// while disabled, which also freezes its gain state, same as
	// bassGuard's own bypass.
	var out float64
	for i := 0; i < 4; i++ {
		y := raw[i] * radarSysGain * radarCompInVolGain[i]
		if m.enabled {
			y *= m.comp[i].levelGain(y)
		}
		y *= radarLimInVolGain[i]
		if m.enabled {
			y *= m.lim[i].levelGain(y)
		}
		out += y
	}
	return out
}

func (m *radarMultiband) reset() {
	for i := range m.lp1 {
		m.lp1[i].reset()
		m.hp1[i].reset()
	}
	for i := range m.lp2 {
		m.lp2[i].reset()
		m.hp2[i].reset()
	}
	for i := range m.lp2c {
		m.lp2c[i].reset()
		m.hp2c[i].reset()
	}
	for i := range m.lp3 {
		m.lp3[i].reset()
		m.hp3[i].reset()
	}
	for i := range m.lp3c1 {
		m.lp3c1[i].reset()
		m.hp3c1[i].reset()
	}
	for i := range m.lp3c2 {
		m.lp3c2[i].reset()
		m.hp3c2[i].reset()
	}
	for i := range m.comp {
		m.comp[i].reset()
		m.lim[i].reset()
	}
}

// setEnabled/setFloorDb/takeMaxReductionDb complete bassStage — see
// chain.go.
func (m *radarMultiband) setEnabled(enabled bool) { m.enabled = enabled }

// setFloorDb reaches band 1's floor only — the one dashboard control
// Radar's guard has (bassGuardDb), same meaning as before this class
// existed. Bands 2-4 have no control, same reasoning as the limiter
// override: there is nothing today to leave untouched.
func (m *radarMultiband) setFloorDb(floorDb float64) { m.comp[0].floorDb = floorDb }

func (m *radarMultiband) takeMaxReductionDb() float64 {
	var worst float64
	for i := 0; i < 4; i++ {
		if m.comp[i].maxReductionDb > worst {
			worst = m.comp[i].maxReductionDb
		}
		if m.lim[i].maxReductionDb > worst {
			worst = m.lim[i].maxReductionDb
		}
		m.comp[i].maxReductionDb = 0
		m.lim[i].maxReductionDb = 0
	}
	return worst
}
