package outchain

import (
	"math"
	"testing"
)

// Pins the table against silent drift — same reason em_mbc.py's own pin
// exists. Band 1's ratio/threshold must also agree with
// bassRatio/radarBassThresholdDb, the same measured band read twice.
func TestRadarBandsMatchItsOwnMeasuredConfiguration(t *testing.T) {
	want := [4]radarBandSpec{
		{20.0, -25.0, -40.0, 0.0, -12.0, 200.0, 0.0},
		{10.0, -18.0, -40.0, 0.0, -12.0, 80.0, 0.0},
		{3.0, -15.0, -40.0, 3.0, -4.0, 20.0, 3.0},
		{2.0, -10.0, -40.0, 3.0, -3.0, 20.0, 0.0},
	}
	if radarBands != want {
		t.Errorf("radarBands = %+v, want %+v", radarBands, want)
	}
	if radarBands[0].compRatio != bassRatio {
		t.Errorf("band 1 ratio = %g, want bassRatio %g", radarBands[0].compRatio, bassRatio)
	}
	if radarBands[0].compThresholdDb != radarBassThresholdDb {
		t.Errorf("band 1 threshold = %g, want radarBassThresholdDb %g",
			radarBands[0].compThresholdDb, radarBassThresholdDb)
	}
	if radarMbclInVolDb != 4.0 {
		t.Errorf("radarMbclInVolDb = %g, want 4.0", radarMbclInVolDb)
	}
}

// A pure limiter is bandGain with an infinite ratio and a -infinite floor.
// Pins that the reuse actually degenerates correctly rather than producing
// NaN or a panic from a divide involving infinities.
func TestBandLimiterDegeneratesCorrectly(t *testing.T) {
	l := newBandLimiter(-6.0, 20.0, 48000)
	if l.ratio != math.Inf(1) {
		t.Fatalf("ratio = %g, want +Inf", l.ratio)
	}
	if l.floorDb != math.Inf(-1) {
		t.Fatalf("floorDb = %g, want -Inf", l.floorDb)
	}
	// 0dBFS peak against a -6dB threshold: a true limiter pins the output
	// exactly at the threshold (6dB of reduction), whatever happens after.
	g := l.levelGain(fullScale - 1) // just under the ceiling, well over -6dB
	gotDb := 20 * math.Log10(g)
	if math.Abs(gotDb-(-6.0)) > 0.01 {
		t.Errorf("gain = %gdB, want -6dB (hard limit at threshold)", gotDb)
	}
	if math.IsNaN(g) || math.IsInf(g, 0) {
		t.Fatalf("gain = %g, want a finite number", g)
	}
}

// A signal entirely below every band's threshold must pass through at
// unity — the network's crossover alone (no law engaging) still has the
// fixed system gain and bands 3/4's input trims, so "no law" is not the
// same claim as "no change in level" (see the Python RadarMultiband's own
// equivalent test and process's comment on why those trims are
// unconditional).
func TestRadarMultibandQuietSignalEngagesNoLaw(t *testing.T) {
	m := newRadarMultiband(48000)
	m.enabled = true
	// Stock seeds every band's level at 0.01 (-20dB power), over band 1's
	// -25dB threshold, so a fresh compressor reduces briefly while that
	// level falls and its gain eases back (~654ms). That is stock's own
	// start; the claim is about the steady state after it.
	for i := 0; i < 6*48000; i++ {
		if i == 5*48000 {
			m.takeMaxReductionDb()
		}
		// Very quiet relative to every threshold (-40dB floor, -25..-10dB
		// thresholds): 1000Hz at roughly -60dBFS.
		x := 32.0 * math.Sin(2*math.Pi*1000*float64(i)/48000)
		m.step(x)
	}
	if r := m.takeMaxReductionDb(); r > 0.01 {
		t.Errorf("max reduction = %gdB on a signal well under every threshold, want ~0", r)
	}
}

// A loud low tone must engage band 1's law (compressor + its own limiter)
// hard, mirroring em_mbc.py's own RadarMultiband smoke test.
func TestRadarMultibandLoudBassEngagesBandOne(t *testing.T) {
	m := newRadarMultiband(48000)
	m.enabled = true
	fs := 48000.0
	var peakOut float64
	for i := 0; i < int(fs*2); i++ {
		x := 30000.0 * math.Sin(2*math.Pi*50*float64(i)/fs)
		y := m.step(x)
		if a := math.Abs(y); a > peakOut {
			peakOut = a
		}
	}
	if r := m.comp[0].takeMaxReductionDb(); r < 10.0 {
		t.Errorf("band 1 compressor max reduction = %gdB, want well over 10dB on a loud 50Hz tone", r)
	}
	if peakOut >= 30000.0 {
		t.Errorf("peak out = %g, want it reduced from the 30000 input", peakOut)
	}
}

// Disabling must freeze every stage's gain state rather than resetting it
// — toggling back on should resume from where it left off, not from
// unity, same contract as bassGuard/limiter bypass.
func TestRadarMultibandDisableFreezesGainState(t *testing.T) {
	m := newRadarMultiband(48000)
	m.enabled = true
	for i := 0; i < 4800; i++ {
		m.step(30000.0 * math.Sin(2*math.Pi*50*float64(i)/48000))
	}
	gainBefore := m.comp[0].gain
	if gainBefore == 1 {
		t.Fatal("band 1 compressor never engaged — nothing to freeze")
	}
	m.setEnabled(false)
	for i := 0; i < 100; i++ {
		m.step(0)
	}
	if m.comp[0].gain != gainBefore {
		t.Errorf("gain drifted from %g to %g while disabled, want frozen",
			gainBefore, m.comp[0].gain)
	}
}

// setFloorDb must reach band 1 only — bands 2-4 have no dashboard control,
// same reasoning as the limiter override.
func TestRadarMultibandSetFloorDbOnlyTouchesBandOne(t *testing.T) {
	m := newRadarMultiband(48000)
	want2 := m.comp[1].floorDb
	m.setFloorDb(-10.0)
	if m.comp[0].floorDb != -10.0 {
		t.Errorf("comp[0].floorDb = %g, want -10", m.comp[0].floorDb)
	}
	if m.comp[1].floorDb != want2 {
		t.Errorf("comp[1].floorDb changed to %g, want unchanged %g", m.comp[1].floorDb, want2)
	}
}

// reset must actually clear every one of the twelve filter pairs' state —
// a stale biquad left non-zero would leave an audible tail on the device
// at every reactivation.
func TestRadarMultibandResetClearsEveryFilter(t *testing.T) {
	m := newRadarMultiband(48000)
	m.enabled = true
	for i := 0; i < 4800; i++ {
		m.step(30000.0 * math.Sin(2*math.Pi*500*float64(i)/48000))
	}
	m.reset()
	pairs := []*[2]biquad{
		&m.lp1, &m.hp1, &m.lp2, &m.hp2, &m.lp2c, &m.hp2c,
		&m.lp3, &m.hp3, &m.lp3c1, &m.hp3c1, &m.lp3c2, &m.hp3c2,
	}
	for pi, p := range pairs {
		for i, bq := range p {
			if bq.z1 != 0 || bq.z2 != 0 {
				t.Errorf("pair %d stage %d not cleared: z1=%g z2=%g", pi, i, bq.z1, bq.z2)
			}
		}
	}
	for i := range m.comp {
		if m.comp[i].gain != 1 || m.comp[i].level != compInit {
			t.Errorf("comp[%d] gain=%g level=%g after reset, want stock's 1 and 0.01", i, m.comp[i].gain, m.comp[i].level)
		}
		if m.lim[i].gainDb != 0 {
			t.Errorf("lim[%d].gainDb = %g after reset, want 0", i, m.lim[i].gainDb)
		}
	}
}
