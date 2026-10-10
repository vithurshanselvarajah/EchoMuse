# Radar audio: how close EchoMuse is to stock

**Short answer:** with the settings below, every stage of Radar's playback
chain now does what stock FireOS does, using stock's own values and the
algorithms read out of stock's binaries. What remains different is listed
under [What still differs](#what-still-differs); none of it is expected to
be audible, but nobody has yet compared the two by ear or by measurement.

Applies to the Echo 2nd gen (Radar, `device_type_id` `A7WXQPH584YP`) from
firmware `a4c741b` and the controller of the same commit.

## Settings that give stock's sound

| Dashboard setting (Config → Playback) | Set to | Why |
|---|---|---|
| Radar's own stock EQ curve | **On** | Turns on the FIR, ParametricEQ and OutputTrim |
| EQ bands | **Flat** | Stock has no user EQ in this path; any band adds to stock's curve |
| Speech boost | **Off** | Not part of stock |
| Speaker protection (bass guard) | **On** | On Radar this switch IS stock's MBCL compressor bands |
| Limiter | **On** | MBCL's full-band limiter |

The guard depth slider reaches only band 1's floor. Stock's is −40dB, our
default −30dB; band 1 cannot reach either at any real level, so the slider
makes no audible difference.

> The older advice in [radar-speaker.md](radar-speaker.md) to turn Speaker
> protection off predates this port. It was written when the guard was
> biscuit's tuning. On Radar it is now stock's MBCL, and turning it off moves
> away from stock.

## The chain, stage by stage

Stock's order comes from `AFE.cfg` (`Playback.Algorithms`). Ours runs in the
same order, on the Echo itself (`device/internal/outchain`), with the
controller's Python (`em_eq`, `em_mbc`, `em_limiter`) as the reference it is
tested against sample for sample.

```
volume ─► EQ FIR (by volume) ─► ParametricEQ ─► MBCL ─► OutputTrim ─► DAC (unity)
                                                 │
            +4dB ─► 4-band split ─► per band: compressor ─► limiter ─► sum ─► full-band limiter
```

| Stage | Stock | EchoMuse | Source |
|---|---|---|---|
| Volume steps | 30 Alexa steps, value per step from `VolumeCurves.xml` | Same 30 steps on the buttons, same table for the HA slider | `VolumeCurves.xml`, `/system/bin/mixer` |
| Volume law | Mixer table, 0.5dB per step, 127 = 0dB | Same table | `/system/bin/mixer` |
| Where volume is applied | Before the AFE | Before the chain | `libaudioCtrl.so` |
| DAC | Never written, stays at 0dB (127) | Held at 127 | `audio_device.xml` |
| Codec speaker filter | 117-byte biquad set from `audio_device.xml` | Read from the Echo's own file at start-up | `audio_device.xml` |
| EQ FIR | `EQ_50/60/70/80/100.cfg`, first boundary ≥ volume value | Same files, same rule | `AFE.cfg`, `libasp.so` |
| ParametricEQ | Low shelf 150Hz +5dB, peak 80Hz +2dB, Q 0.9 | Same | `ParametricEQ.cfg` |
| MBCL input gain | +4dB | +4dB | `MBCL.cfg` |
| Crossovers | 70 / 200 / 3250Hz, Butterworth LP/HP + allpass | Linkwitz-Riley with allpass compensation (the same filters) | `MBCL.cfg`, `libasp.so` |
| Band compressors | See below | Ported | `libasp.so` |
| Band and full-band limiters | See below | Ported | `libasp.so` |
| OutputTrim | +3dB after MBCL | Same | `AFE.cfg` |
| Music AVL | Off by default | Not ported | `libaudioCtrl.so` |

### How the volume picks the EQ file

`libaudioCtrl` looks up the Alexa step's 0–100 value in `VolumeCurves.xml`,
and the mixer hands that value to `libasp` as command 5
(`executeAspCommandWithIntInput(5, vol)`). `libasp` passes it to the
Equalizer FIR module, which plays the first file whose boundary in
`[50, 60, 70, 80, 100]` is at or above it. The files are a loudness
compensation: the 80Hz boost is +10.1dB on `EQ_50`, +5.4dB on `EQ_80` and
+1.4dB on `EQ_100`, so the bass backs off as the volume rises.

### MBCL bands (`MBCL.cfg`)

| Band | Range | Comp in | Ratio | Threshold | Floor | Lim in | Lim threshold | Release (config → runs at) |
|---|---|---|---|---|---|---|---|---|
| 1 | 0–70Hz | 0dB | 20:1 | −25dB | −40dB | 0dB | −12dB | 200ms → 200ms |
| 2 | 70–200Hz | 0dB | 10:1 | −18dB | −40dB | 0dB | −12dB | 80ms → 180ms |
| 3 | 200–3250Hz | +3dB | 3:1 | −15dB | −40dB | +3dB | −4dB | 20ms → 180ms |
| 4 | 3250Hz+ | +3dB | 2:1 | −10dB | −40dB | 0dB | −3dB | 20ms → 180ms |
| Full band | | | | | | 0dB | −3dB | 20ms → 180ms |

### The compressor (`libasp.so` 0xecea0 / 0xed168)

Stock's compressor works in 1ms blocks (48 samples) and reacts slowly, which
keeps it from pumping on kick drums:

1. `y = x × input gain`; block power `P = mean(y²)` (full scale = 1).
2. **Gate.** A fast average `A` of the power, and a slow floor `B` under it.
   The level only moves while `A` is more than 38.4dB above `B`, so it holds
   through pauses.
3. **Level.** Eases toward `P`: rising takes about 43ms, falling about 435ms.
4. **Gain.** Unity at or under the threshold; above it,
   `(1 − 1/ratio) × (threshold − level)` dB, never deeper than the floor.
5. **Gain smoothing.** Eases toward that gain over about 654ms, both ways.
6. **Look-ahead.** The gain is applied to audio delayed 16ms (768 samples).

It starts every stream with the level at 0.01 (−20dB), which is above band
1's threshold, so the bass eases in over the first half second. Stock does
the same.

### The limiter (`libasp.so` 0x8d600 / 0x8d96c)

1. Look-ahead of 2ms (96 samples).
2. When a sample would cross the threshold, the gain drops to put it exactly
   on it, and the 96 samples already queued are faded down with it. The cut
   starts before the peak arrives.
3. Holds for 20 samples, then releases in a straight line back to unity.
4. The release is clamped to 180–400ms, so the config's 20ms and 80ms both
   run at 180ms.

## What still differs

| Difference | Size | Audible? |
|---|---|---|
| The compressor adds 1ms of delay on top of stock's 16ms, because our audio periods do not line up with its 1ms blocks | 48 samples, same on every band | No |
| Changing EQ file: stock switches instantly, we crossfade over one period (43ms) | | No; stock's switch could click |
| Maths in 64-bit floats; stock uses 32-bit and an approximate divide | Last-digit rounding | No |
| After a silent period the Echo resets the chain to stock's start-up state; stock may keep its state between streams | Bass eases in again after a pause | Possibly, on the first half second after a pause |
| Speech (TTS) uses stock's music volume; stock's own speech curve is 11dB quieter at maximum | Up to 11dB on replies | Yes, replies are louder than on stock |
| Music AVL (stock's automatic volume levelling) is not ported | Off by default on stock | No, at stock's default |
| The controller-side chain (firmware without `output_chain`) has no volume, so it always plays `EQ_50` | Every Radar firmware has `output_chain` | No |

Not checked yet: a measurement or a listening comparison against a stock
Radar playing the same track at the same volume step. That is the test that
would settle "exactly like stock".

## Where it lives

| What | Python (reference) | Go (on the Echo) |
|---|---|---|
| Volume table and EQ file choice | `controller/em_volume.py`, `em_eq.py` | `outchain/eqfir.go`, `internal/server/volume.go` |
| FIR, ParametricEQ, OutputTrim | `em_eq.py`, `radar_eq_banded.json` | `outchain/chain.go`, `eqfir.go` |
| MBCL bands and crossovers | `em_mbc.py` (`RadarMultiband`) | `outchain/radarmbc.go` |
| Compressor | `em_mbc.py` (`StockCompressor`) | `outchain/stockcomp.go` |
| Limiter | `em_limiter.py` (`StockLimiter`) | `outchain/stocklimiter.go` |
| Codec filter start-up | | `bindings/speaker/radar.go` |

`device/internal/outchain/testdata/gen_vectors.py` renders the reference
output; `TestMatchesControllerChain` holds the Go to it. The three Radar cases
currently match with 0 samples different.

No stock binary is shipped. Two sets of stock data are: the five FIR curves,
converted from stock's `EQ_*.cfg` into `radar_eq_banded.json`, and the
numbers in the tables above. The codec filter is not shipped; it is read
from the Echo's own `/system` at start-up.
