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

// radarDacUnity is Radar's own DAC digital-volume baseline — NOT 127, and
// deliberately not the shared dacUnity constant (pcm_speaker.go), which is
// biscuit's own measured value and is never reached on this path at all
// (Init branches to startRadarOutput before dacUnity is ever read).
//
// 127 was carried over from biscuit on the assumption that this control's
// "0dB unity" point is the same across both boards' codecs. It is not: on
// biscuit, indexes above 127 apply positive digital gain to near-full-scale
// PCM and measurably clip (device/internal/server/volume.go). On Radar,
// stock firmware runs this exact control permanently at 255 and does all
// its own volume control in AudioFlinger software instead — confirmed from
// a Radar unit's own stock audio_policy_configuration.xml and
// default_volume_tables.xml — so 127 was never Radar's unity point, just a
// borrowed number nobody had reason to question until the speaker measured
// quieter than stock at every volume level.
//
// 145, not 255: measured directly on hardware (owner's own unit, tinymix
// 'PCM Playback Volume' <n> while audio played) — clean at 150, still clean
// one step up at 175 but the device browned out after ~1s there,
// consistent with the test rig's power supply rather than the codec
// (current draw rising with level, not a codec fault). 145 is the owner's
// own choice, by ear against a stock Echo side by side: 140 (an earlier
// deliberately-conservative pick) sounded too quiet once compared
// properly, 148 overshot it back the other way, and 145 is where it
// settled. 150 itself was fine but left no margin at all under the 175
// brownout. Headroom above 150 likely exists and is explicitly left
// unclaimed until it's verified on better-provisioned hardware.
const radarDacUnity = "145"

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
