package speaker

import (
	"encoding/xml"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/wilbowes/EchoMuse/internal/bindings/mixer"
)

const radarMute = "MFP Gpio Mute"

// Read the device's own speaker calibration, not the headphone profile, and
// never redistribute stock files in a build. FireOS's HAL normally loads this;
// emOS has no HAL. An all-zero profile leaves Radar silent (#535).
func radarSpeakerProfile(data []byte) ([]string, error) {
	var doc struct {
		XMLName xml.Name `xml:"mixercontrol"`
		Paths   []struct {
			Name     string `xml:"name,attr"`
			Value    string `xml:"value,attr"`
			Controls []struct {
				Name  string `xml:"name,attr"`
				Value string `xml:"value,attr"`
			} `xml:"kctl"`
		} `xml:"path"`
	}
	if err := xml.Unmarshal(data, &doc); err != nil {
		return nil, err
	}
	var profile []string
	for _, path := range doc.Paths {
		if path.Name != "ext_speaker_output" || path.Value != "turnon" {
			continue
		}
		for _, ctl := range path.Controls {
			if ctl.Name != "biquad coefficients" {
				continue
			}
			if profile != nil {
				return nil, fmt.Errorf("duplicate Radar speaker profile")
			}
			profile = strings.Fields(ctl.Value)
		}
	}
	if len(profile) != 117 {
		return nil, fmt.Errorf("Radar speaker profile: expected 117 bytes, got %d", len(profile))
	}
	nonzero := false
	for _, s := range profile {
		v, err := strconv.ParseUint(s, 10, 8)
		if err != nil {
			return nil, fmt.Errorf("Radar speaker profile: invalid byte %q", s)
		}
		nonzero = nonzero || v != 0
	}
	if !nonzero {
		return nil, fmt.Errorf("Radar speaker profile is all zero")
	}
	return profile, nil
}

// Keep physical mute asserted before any codec/clock changes. Failure must
// leave the speaker muted, never release an unconfigured DAC at full gain.
func prepareRadarSpeaker(path string) error {
	if err := mixer.Set(radarMute, "On"); err != nil {
		return err
	}
	if err := mixer.Set(mixer.PlaybackVolume, "0"); err != nil {
		return err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	profile, err := radarSpeakerProfile(data)
	if err != nil {
		return err
	}
	for _, w := range []mixerWrite{
		{mixer.SpeakerAmp, []string{"Off"}},
		{"biquad coefficients", profile},
		{"Audio_I2S1_Setting", []string{"On"}},
		{"HP DAC Playback Switch", []string{"1", "1"}},
		{"Audio_DacMux_Setting", []string{"Off"}},
		{"DRC Control", []string{"Disabled"}},
		{mixer.HPDriverGain, []string{"6", "6"}},
	} {
		if err := mixer.Set(w.Ctl, w.Args...); err != nil {
			return err
		}
	}
	return nil
}

// radarDacUnity is Radar's DAC digital-volume baseline: 127, the
// tlv320aic32x4's 0dB point, which is what stock runs it at. Kept separate
// from dacUnity (pcm_speaker.go) because Init branches to startRadarOutput
// before that is read, not because the value differs.
//
// It was 150, then 140, then 145 (38a874d, 8d33d2b, 292b168), and the
// reasoning behind that was wrong in two places, both checkable in
// device/tools/radar_dump:
//
//   - Radar's playback DAC is NOT a different chip. It is the same
//     tlv320aic32x4 as biscuit (2-0018), whose "PCM Playback Volume" runs
//     0..175 in 0.5dB steps with 0dB at 127 — 175 being the control's
//     maximum is also why that was where the bench "browned out".
//   - Stock does NOT run it at 255. 255 appears only in mixer_paths.xml,
//     a generic MediaTek file whose other controls (Audio Amp Playback
//     Volume, LINEOUT Mux, HPOUT Mux, I2S O03_O04 Switch, AIF TX Mux) do
//     not exist on this codec at all, and 255 is outside the control's
//     range. The HAL's real file, audio_device.xml, never writes it, and
//     the stock tinymix dump reads 127 127. Stock attenuates in
//     AudioFlinger and leaves the DAC at its reset 0dB, exactly as on
//     biscuit (device/CLAUDE.md, volume section).
//
// 145 still sounded right by ear because it was making up for gain the
// output chain did not have yet: MBCL's +4dB system gain and band 3/4's
// +3dB trims (8239c66), OutputTrim's +3dB and ParametricEQ (f21b20f), and
// the volume-matched EQ file (EQ_100 is 2dB hotter at 1kHz than EQ_50).
// With those ported, the chain's output measured 8-13dB hotter in RMS than
// the chain 145 was tuned against, peaks reach 0dBFS at full volume, and
// 145 would add +9dB on top INSIDE the DAC — where biscuit measured THD of
// 65% at 153 on near-full-scale input. Any level difference against a
// stock Echo from here belongs in the chain, which is in software and
// measurable, never in DAC gain above 0dB.
const radarDacUnity = "127"

// radarDacUnityLevel is radarDacUnity as an int, for the ramp loop below —
// ONE source for both, deliberately: the loop bound used to be a second,
// separately-maintained literal ("150"), which is exactly how this value
// drifted out of sync with itself the first time this was tuned.
var radarDacUnityLevel = func() int {
	n, err := strconv.Atoi(radarDacUnity)
	if err != nil {
		panic("radarDacUnity must parse as an int: " + err.Error())
	}
	return n
}()

// The measured quiet-start sequence from the Radar bench: clock silence for
// three seconds at gain zero, release physical mute, wait four seconds, then
// ramp the DAC. User volume remains in software; Init has not returned yet,
// so no caller can enqueue audio or cues during this sequence.
func unmuteRadarSpeaker(wait func(time.Duration) error) (err error) {
	defer func() {
		if err != nil {
			mixer.Set(radarMute, "On")
			mixer.Set(mixer.PlaybackVolume, "0")
		}
	}()
	if err = mixer.Set(mixer.SpeakerAmp, "On"); err != nil {
		return err
	}
	if err = wait(3 * time.Second); err != nil {
		return err
	}
	if err = mixer.Set(mixer.PlaybackVolume, "0"); err != nil {
		return err
	}
	if err = mixer.Set(radarMute, "Off"); err != nil {
		return err
	}
	if err = wait(4 * time.Second); err != nil {
		return err
	}
	for v := 10; v < radarDacUnityLevel; v += 10 {
		if err = mixer.Set(mixer.PlaybackVolume, strconv.Itoa(v)); err != nil {
			return err
		}
		if err = wait(50 * time.Millisecond); err != nil {
			return err
		}
	}
	return mixer.Set(mixer.PlaybackVolume, radarDacUnity)
}
