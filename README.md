# temple-snr-repro

Reproduce one number yourself, from public data: **the signal-to-noise ratio of contrast-free blood on a single ultrasound receive element, per transmitted pulse, through the adult temporal bone at about 40 mm depth.**

**Result on all 20 recordings (10 subjects):** −22.4 dB at 37.5–42.5 mm (90 % CI −24.8 to −18.9 dB). The pre-set precision rule was **not** met at 40–45 mm (CI 6.45 dB wide, limit 6.0 dB), so the formal verdict is **FAIL** (the script prints `FAIL`; the failing criterion is P2 width, shown as `FAIL(WIDE)` on the P2 line). Details are in [Full result](#full-result-all-20-recordings).

**Noise-band check:** fast blood does reach the 1500–2000 Hz noise band (pointed out by Maxence Reberol, Resolve Stroke); measured, it moves the result by about +0.1 dB, to about −22.2 dB. See [Check](#check-blood-in-the-noise-band).

**Second result, same data:** whitening the correlated receive-channel noise before beamforming lowers the noise floor of contrast-free power Doppler maps by about 1 dB and raises vessel contrast slightly (median CNR gain +0.16 in 8 held-out subjects; pre-registered verdict **PASS**, reproduced independently). It is a re-test on recordings already seen, not an independent replication. See [Second result](#second-result-receive-noise-whitening-1-db) and `whitening/`.

## Data

Open-H-RF, Resolve Stroke clinical recordings (Hugging Face `nvidia/Open-H-RF`, folder `resolvestroke/clinical/`): 20 acquisitions from 10 subjects (`SP01`–`SP10`), 2 per subject. What the sources say about the subjects:
- **Dataset card** (`resolvestroke/clinical/README.md`): "10 human subjects", acquired in the SCULPT study (NCT07324421), a monocentric trial at CHU Gui de Chauliac, Montpellier, "comparing cerebral perfusion ... with routine perfusion CT in ICU/CCU patients". The card does not give age, sex or diagnosis.
- **Study registration** (ClinicalTrials.gov NCT07324421): inclusion requires "Patient ≥ 18 years old" and admission to ICU or CCU with an indication for CT perfusion. The listed conditions are neuro-ICU, subarachnoid haemorrhage, cerebral ischaemia and vascular brain injury. **"Adult" therefore comes from the registration, not the dataset card.** No source says that these 10 subjects are stroke patients.
- **Open-H-RF paper** (arXiv:2610.13616) describes these data only as "3D transcranial perfusion of the human brain". It gives no subject details.

Acquisition: imaging through the temporal window with a 2 MHz 32 x 32 matrix probe (SYLVER, 0.5 mm pitch), one diverging-wave transmit per frame at 4000 frames/s, I/Q channel data with 256 receive channels. The card calls these channels "virtual elements" ("receive sub-apertures flattened into 256 virtual elements"). Their stored positions are 256 distinct points on the 0.5 mm grid, so we treat each channel as one receive element. That is our reading, not the card's statement.

Each file has five 1-s clips around a microbubble bolus. We use only the clip **before the bubbles arrive** (`baseline_minus1s`). We also use the first 256 frames of the +10 s clip (bubbles present), but only to check that the "before" clip is free of bubbles.

Nothing here depends on any other repository. Four files do the work:

| file | what it does |
|---|---|
| `get_data.py` | downloads only the byte ranges holding the frames we need (about 1 GB per recording instead of 5.3 GB), verifies them, and can resume after an interruption. It lists all 20 recording names |
| `temple_snr.py` | decodes the frames, applies the depth gates, notches the artefact lines, applies the clutter filter, estimates noise, and computes SNR per element, gate, recording and subject. It prints the tables and saves one figure. All analysis choices are in one `CONFIG` block at the top of the file |
| `temple_snr.ipynb` | calls both scripts and explains each step in plain language |
| `LICENSE` | MIT licence for the code |
| `example_output/` | the figure, JSON and log from our quick run (below), for comparison with yours |
| `whitening/` | the second result: `whitening.py` (receive-noise whitening of 3-D power Doppler maps) and its `example_output/`. See [Second result](#second-result-receive-noise-whitening-1-db) |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or requirements-notebook.txt to use the notebook
python get_data.py --n 4                 # 4 recordings, ~3.8 GB download, resumable
python temple_snr.py --selftest          # optional: synthetic data with known SNR (-22 dB) must come back within 0.5 dB
python temple_snr.py                     # prints the tables, writes results/temple_snr.png, results/temple_snr_results.json
                                         # and results/per_element_snr_linear.npz (per-element SNR, linear)
```

Recording sets:

| command | recordings | download | what you should get (P1, estimator E) |
|---|---|---|---|
| `get_data.py --n 4` (quick run) | first recording of SP01–SP04 | ~3.8 GB | −21.7 dB, 3 subjects, FAIL(FEW); the table is below |
| `get_data.py --n 10` | first recording of every subject | ~9.5 GB | −22.1 dB, 9 subjects (SP04-Left-1 fails the contamination check), CI width 5.0 dB; P2 −24.5 dB, CI width 7.7 dB |
| `get_data.py --all` (= `--n 20`) | **all 20, the full study** | ~19 GB | −22.4 dB, 10 subjects, 19 recordings; see [Full result](#full-result-all-20-recordings) |

`--n K` takes the first K names of the list in `get_data.py`: the 10 "first" recordings, then the 10 "second" ones (SP01-Left-2, SP02-Left-2, SP03-Right, SP04-Left-2, SP05-Right, SP06-Right, SP07-Right, SP08-Right, SP09-Left-2, SP10-Right). For explicit recordings, use `--recordings SP05-Left SP03-Right ...`. Unknown names and `--n` outside 1–20 are rejected. `temple_snr.py` analyses every recording in the data folder; `temple_snr.py --recordings ...` restricts it to a subset. To put the data somewhere else, use `--data-dir` for both scripts. Notebook: `pip install -r requirements-notebook.txt && jupyter notebook temple_snr.ipynb`.

## Requirements (exact versions tested)

We tested with CPython 3.14.4 on macOS (Apple silicon), in fresh virtual environments created only from these files.

`requirements.txt` (enough for both scripts):
```
numpy==2.5.4
scipy==1.18.1
h5py==3.16.0
hdf5plugin==7.1.0      # Blosc/zstd decoder used by the dataset
matplotlib==3.11.2
```
`requirements-notebook.txt` adds `notebook==7.6.3`, `ipykernel==7.4.0` and `nbconvert==7.17.2`. The full resolved environment of the quick run is in `example_output/pip_freeze_quick_run.txt`.

## Runtime

Quick run (4 recordings), measured on 2026-10-21 (our run plus one independent rerun) on a MacBook (Apple silicon) over a ~40 MB/s connection, using a fresh venv and an empty data folder:

| step | time |
|---|---|
| `pip install -r requirements.txt` | 10–11 s |
| `get_data.py --n 4` (3.77 GB, 4 x 4256 chunks, all checks) | 216–249 s |
| `temple_snr.py --selftest` | 18–46 s |
| `temple_snr.py` (4 recordings, incl. 4 x 4000-replicate bootstraps) | 29–40 s |
| **total** | **about 5 min** |
| notebook, executed end-to-end with `jupyter nbconvert --execute` in a second fresh venv (data already present) | 46–66 s |

The download is almost all of the time, so it scales with your bandwidth. Analysis takes about 6–11 s per recording and peaks at about 2.1 GB of RAM. For all 20 recordings (`--all`), a cold run on 2026-10-22 (fresh clone and venv, MacBook, Apple silicon) downloaded 18.8 GB in 20.1 min (about 16 MB/s effective, including chunk verification) and analysed all 20 in about 2 min: about 24 min in total, reproducing every number below exactly.

## What the analysis computes (estimator E, primary)

For each recording:
1. **Decode** the 4000 pre-bolus frames (int16 I/Q, 320 fast-time samples x 256 elements) and divide by the scanner's TGC curve.
2. **Gate.** For element *i* at (x, y) and depth z, the two-way time is t = [sqrt(x² + y² + (z + 22 mm)²) − 22 mm + z] / c. Here 22 mm is the virtual source behind the probe (from the file's `focus_distances`). The headline gate is **P1 = 37.5–42.5 mm**, co-primary **P2 = 40–45 mm**, and six 5-mm gates span 30–60 mm. A (sample, element) cell is dropped if any frame clips (|I| or |Q| ≥ 32000). An element is dropped if more than half of its cells are dropped.
3. **Slow-time spectrum** of each cell: remove the mean over frames, apply a 4-term Blackman–Harris window, then a 4000-point FFT, giving 1 Hz bins. Each element's spectrum is the average over its gate samples.
4. **Notch artefact lines.** Narrow scanner lines sit at exactly k·PRF/4 (1000, 2000, 3000 ≡ −1000 Hz). They are not blood. Bins within ±8 Hz of these lines are removed from every average.
5. **Clutter filter.** The wall band |f| < 100 Hz holds skull and tissue echo, which is thousands of times stronger than blood. It is excluded.
6. **Noise.** N = mean spectral density in 1500 ≤ |f| ≤ 2000 Hz. Blood faster than about 0.57 m/s along the beam (for example in the MCA) does reach this band, so it is not pure noise; we measured how much blood is in it and the effect on the result is about +0.1 dB. See [Check: blood in the noise band](#check-blood-in-the-noise-band).
7. **SNR** per element = (D − N)·2800 / (N·4000), where D = mean density in 100 ≤ |f| < 1500 Hz. This is the blood-band power above the noise floor divided by the noise power per transmit. The recording value is the median over valid elements.

**Cross-check X** takes an independent route. It subtracts the exact period-4 component per cell, which notches k·PRF/4 in the time domain. It then applies a 4th-order Butterworth high-pass at 100 Hz (`sosfiltfilt`) and drops 200 frames at each end. Noise comes from the 4th-order temporal difference, which weights noise near PRF/2. SNR = (P − κσ²)/σ². κ and the difference gain come from the filter's frequency response, not from a simulation (see the notes).

**Contamination check C.** In the 30–60 mm gate, E is computed on the contrast clip (256 frames, ±4-bin notch) and on the pre-bolus clip. The pre-bolus value is floored at 2 × its bootstrap standard error (500 resamples of element columns) or 10^−3.5, whichever is larger. C = 10·log10(contrast / pre-bolus) at gate shifts of −1, 0 and +1 sample. A recording is used only if min C ≥ 6 dB, so the bubbles must clearly raise the blood-band power over the "before" clip.

**Aggregation.** One value per subject is the linear mean of that subject's included recordings (with `--all`, usually 2). The population value is the median over subjects. The 90 % interval comes from a two-stage cluster bootstrap with 4000 replicates: subjects are resampled, then the 32 element columns of 8 elements are resampled within each recording. `temple_snr.py` prints a per-recording table, a per-subject table, and a verdict per primary gate and overall.

**Pass rule.** All of these definitions, including the pass rule, were written down and frozen before the 20-recording analysis was run. Each primary gate (P1 and P2) needs ≥ 8 of 10 subjects, a 90 % CI width < 6 dB, and |E − X| ≤ 3 dB, and the overall verdict is PASS only if both gates pass. **A 4-recording quick run therefore always reports FAIL(FEW). This is expected and is not a finding.**

Our full analysis also ran sensitivity checks that this package does not reproduce: gate shift, wall frequency, notch width, noise band, no notch, and subsets. It also ran flatness, line and stationarity diagnostics. The values that matter for interpretation are quoted below.

## Full result (all 20 recordings)

Our full analysis applied the same frozen definitions to all 20 recordings, using a separate implementation. **19 of 20 recordings passed the contamination check**: SP04-Left-1 failed (C_min 3.7 dB), and SP04 is still included through SP04-Left-2. All 10 subjects are therefore included.

| gate | subjects (recordings) | E median | 90 % CI | CI width | X median | verdict |
|---|---|---|---|---|---|---|
| P1 37.5–42.5 mm | 10 (19) | **−22.4 dB** | −24.8 to −18.9 dB | 5.90 dB | −22.1 dB | PASS |
| P2 40–45 mm | 10 (19) | **−22.6 dB** | −25.5 to −19.1 dB | **6.45 dB** (prints as 6.5) | −23.2 dB | FAIL(WIDE) |

**Overall verdict under the pre-set rule: FAIL** (P2 is `FAIL(WIDE)`: its CI, printed as 6.5 dB and 6.453 dB in the JSON, exceeded the 6.0 dB limit). All other criteria were met at both gates.
- **The P1 pass is marginal.** Its CI clears the limit by only 0.1 dB, and a ±1-sample gate shift makes it 6.2–6.6 dB (WIDE). The fair summary is that the CI width sits at about 6 dB at both depths.
- **The width comes from recording-to-recording variation, not from the estimator.** Two recordings of the same subject differ by an SD of about 5.4 dB at P1. Splitting one recording's elements in half changes the value by only about 1 dB. A narrower CI therefore needs more recordings or subjects, not a better estimator.
- **Subjects differ.** Subject values at P1 span −28.2 to −13.5 dB (5–95 %: −27.4 to −14.7 dB). The two estimators agree within 1.4 dB at every depth. The value falls from about −17 dB at 30–35 mm to about −24 dB at 45–60 mm.
- **Check against this package.** We checked that this package gives the same result in two ways, without running it end-to-end on all 20 downloads. (1) It reproduced 5 of the 20 recordings (the quick-run 4 and SP03-Right) to within 0.1 dB. (2) Its aggregation code, fed the full analysis's per-element values, returns the numbers above exactly.

## Result of our quick run (4 recordings, printed table verbatim)

```
Contrast-free blood SNR, dB per element per transmit (median over valid elements)
recording      Cmin incl  nP1   P1 E   P1 X P1 no-notch   P2 E   P2 X  |   G30   G35   G40   G45   G50   G55
SP01-Left-1    13.7  yes  256  -16.8  -17.9       -16.1  -18.2  -20.3  | -13.8 -14.0 -18.2 -21.3 -23.2 -22.8
SP02-Left-1    11.4  yes  256  -22.1  -23.3       -21.1  -24.5  -25.2  | -17.3 -21.4 -24.5 -25.5 -30.9 -23.4
SP03-Left      21.0  yes  256  -21.7  -20.8       -21.0  -20.3  -19.8  | -10.5 -19.3 -20.3 -16.0 -14.4 -14.0
SP04-Left-1     3.7   no  255  -18.6  -19.5       -16.6  -18.6  -19.6  | -23.3 -19.9 -18.6 -19.2 -21.2 -21.0
Cmin = contamination check, min over gate shifts of contrast/pre-bolus blood-band power, 30-60 mm (dB); included if >= 6.0. '<res' = not above the noise floor.
P1 (37.5, 42.5) mm: E median -21.7 dB, 90% CI [-22.5, -16.7] (width 5.8 dB); X median -20.8 dB; n = 3 subjects; protocol verdict FAIL(FEW)
P2 (40.0, 45.0) mm: E median -20.3 dB, 90% CI [-24.6, -18.1] (width 6.5 dB); X median -20.3 dB; n = 3 subjects; protocol verdict FAIL(FEW+WIDE)
```

The current script also prints a per-subject block before the two summary lines, and an overall verdict line after them. Here each subject has one recording, so the subject values equal the recording values. The overall verdict is FAIL. The figure in `example_output/` was made with an earlier version, whose 9-bin smoothing zero-padded the spectrum edges and drew a false dip near ±2000 Hz. The current version uses reflect padding.

Synthetic self-test with a true SNR of −22.0 dB: E gave −22.0 dB and X gave −21.8 dB on the 30–60 mm gate, which is the gate that is tested. The self-test also prints P1 and P2 values marked "(info, not tested)". On one synthetic recording a 5-mm gate scatters by about ±0.5–1 dB from sampling noise alone, so P1 E = −22.9 dB is expected and not a failure. SP04-Left-1 fails the contamination check (3.7 dB). All 4 × 4256 chunks decoded, were byte-identical when stored, and 32/32 re-fetched chunks per recording matched. `example_output/temple_snr.png` shows SNR against depth and the mean un-notched spectrum, where the ±1000 Hz and −2000 Hz lines are visible.

## What the number means

- The blood echo power (Doppler 100–1500 Hz, above a 100 Hz clutter cut) on **one receive element (channel) of this probe**, for **one transmitted pulse** (unfocused diverging wave), relative to that element's white noise in the same time samples. It is measured at 37.5–45 mm through the intact temporal bone of adult ICU/CCU patients, with no contrast agent. Across 10 subjects it is about **−22 dB**, i.e. the blood echo is roughly 170 times weaker than the noise before any averaging. Individual subjects range from about −28 to −14 dB.
- It is a ratio within the same element and samples, so TGC, receive gain and ADC scaling cancel out.

## What it does NOT mean

- It is **not** an image SNR. Beamforming over elements and averaging over frames add a large gain that is not included.
- It is **not** a property of another probe, frequency, element size, transmit pressure (MI) or sequence. Transferring it to a different design needs a model for those factors. That model is not part of this package, and its uncertainty is likely much larger than that of the measurement.
- It is **not purely blood**. Tissue motion above 100 Hz counts as signal, so in that respect the number is an upper bound. Blood faster than about 0.57 m/s (above 1500 Hz, or aliased) is missed, so in that respect it is a lower bound.
- It **excludes slow blood**. The 100 Hz wall removes all blood slower than about 0.04 m/s along the beam (100 Hz × c / 2f₀), which includes most microvascular and perfusion flow. That is another lower-bound factor. The number also depends on where the wall is set, and much of the excess sits just above it. In the full analysis, a 200 Hz wall lowers P1 by 1.2 dB (to −23.5 dB). Removing the line notch raises it by 2.0 dB (to −20.4 dB). Using only half of the noise band moves it by −0.3 or +0.7 dB.
- It is **not a population estimate from the quick run**. The quick run uses 3 included subjects and only shows the pipeline. Use `--all` for the full study.
- Nothing here is a clinical or device performance claim.

## Notes and known limitations

- **Artefact lines.** Without the notch, the "P1 no-notch" column is 0.7–2.0 dB higher in the four quick-run recordings, and +2.0 dB at population level in the full analysis. The 1000 Hz line falls in the blood band, and the 2000 Hz line inflates the noise floor.
- **Sampling noise.** The measured excess is tiny. At −22 dB, the blood power in the band is about 0.6 % of the total noise power; relative to the noise density in the blood band, D/N − 1 is about 0.9 %. On a single recording a 5-mm gate has about ±0.5 dB of sampling scatter, which is why the self-test passes or fails on a 30-mm gate. For the same reason, κ and g_d for X are computed analytically. A 4000 × 256 white-noise simulation has about 0.2 % sampling error, so simulated constants can shift X by more than 1 dB. With one fixed seed the shift was +1.5 dB; its sign is random, not systematic.
- **Data integrity.** The whole-file sha256 cannot be checked because only parts of each file are downloaded. Integrity rests on decoding, byte identity and re-fetch checks. Your `data/<rec>/manifest.json` lists the sha256 of every chunk, so you can compare it with someone else's.
- **Depth.** The depth-to-sample mapping assumes c = 1540 m/s and the given virtual source. A ±1 sample error is 0.4 mm.
- **Download.** Ctrl-C stops cleanly and keeps finished pieces; rerun the same command to resume.

## Check: blood in the noise band

Maxence Reberol (Resolve Stroke) pointed out that fast blood flow, for example in the middle cerebral artery, reaches 1500–2000 Hz, so that band is not a clean noise reference. He is right about the physics: the vendor's contrast velocity maps show radial speeds above 0.57 m/s in 13–73 % of vessel voxels. The original analysis did not test this. We then measured how much blood actually lands in that band, on all 19 included recordings. κ is the blood density in 1500–2000 Hz as a fraction of the blood density in 100–1500 Hz.

| test | κ | what it shows |
|---|---|---|
| Direction (blood flows one way, receiver noise is symmetric) | 0.033 ± 0.011 | The noise band stays symmetric even where the blood band is strongly one-sided (31 % in SP08-Left) |
| Pulsatility (twenty 50-ms windows over the clip) | 0.012 ± 0.012 at P1 | No heartbeat-linked power in the noise band at 37.5–42.5 mm, while 1100–1400 Hz does pulse (about 0.1), so the test can see blood when it is there |
| Depth profile | 0.029 ± 0.022 | No rise in the noise band at typical MCA depths (45–65 mm) |

**Effect on the result.** With κ ≈ 0.03 the P1 value moves from −22.4 to about **−22.2 dB** (90 % CI about −24.7 to −18.8 dB); even κ = 0.10 gives −21.9 dB. The bias is in the conservative direction (blood in the noise reference makes the reported SNR slightly too low). Other noise references give −21.4 to −22.7 dB. Cross-check X weights noise near 2000 Hz and shares the same ~0.1 dB bias. The whitening result is not materially affected (blood is about 0.1–0.2 % of the power in its 1250–1750 Hz noise band, and a whitening matrix from a recording with no detectable blood works as well).

**Side finding.** The receiver noise in the clinical clips is about 1.5× the noise in the static phantom recorded on the same system (ratio 1.52, flat from 25 to 120 mm, and the same in a recording with no detectable blood). It does not come from blood. Against phantom-level noise the per-channel SNR would be about 1.8 dB higher. The phantom file holds only 128 usable frames, so this comparison is coarse.

The code and per-recording outputs of this check are in the campaign's working files, not in this package.

## Second result: receive-noise whitening (+1 dB)

**Question.** The noise on the 256 receive channels of this probe is not independent. It comes in 32 groups of 8 channels (each group a 2 x 4 patch of the array) that share one noise source; within a group the noise correlation is about 0.3, across groups about 0. Delay-and-sum beamforming assumes independent noise. If that shared noise is removed ("whitened") before beamforming, do contrast-free blood vessels stand out more in a 3-D power Doppler map?

**Answer on 8 held-out subjects (15 recordings): yes, slightly.** The noise floor drops by about 1 dB, the blood signal passes unchanged, and vessel contrast rises a little. The pre-registered verdict is **PASS**, and an independent re-implementation reproduced every per-recording value. This is a re-test on recordings that had already been analysed, **not an independent replication** (see the caveats).

### What was done

Everything below was written down and frozen before the confirmatory run. `whitening/whitening.py` is a standalone re-implementation of it (not the frozen code); its docstring repeats the definitions.

1. **Bands, from the PRF only.** Artefact lines sit at every k·PRF/4 (k = 0 is the tissue clutter at DC). With a guard of PRF/16 = 250 Hz on both sides of every line, two line-free intervals remain: the **blood band B\* = 250–750 Hz** and the **noise-reference band W\* = 1250–1750 Hz** (both signs of f; 124 bins of 8 Hz each).
2. **Data.** The 4000-frame pre-bolus clip, in 8 blocks of 500 frames. Per block: mean removal, Blackman–Harris window, slow-time FFT, keep the B\* and W\* bins.
3. **Noise covariance, per recording.** From the W\* bins, fast-time samples 50–299, each sample normalised by its mean power. It is cross-fitted in time: even blocks are whitened with the covariance of the odd blocks, and vice versa.
4. **Whitening (`full05`).** R = noise correlation matrix, R_a = 0.95 R + 0.05 I, T = D^1/2 R_a^−1/2 D^−1/2, where D holds the channel noise powers (so every channel keeps its noise power and the beamformer weighting is unchanged), scaled so the mean of diag T is 1. The comparison arm is `nominal` (T = I).
5. **Beamforming.** TGC removed, x4 fast-time upsampling, x ← T x, then 3-D delay-and-sum (diverging wave from the 22 mm virtual source, c = 1540 m/s, uniform weights) on a grid of x, y = ±18 mm (1 mm) and z = 30–60 mm (0.5 mm), 83 509 voxels. Per voxel: mean power in B\* and in W\*.
6. **Score.** S\* = ⟨B\*⟩/⟨W\*⟩ − 1 per voxel (0 = noise only), smoothed with a 1-voxel Gaussian. The **vessel mask** is the top 10 % of the vendor reference volume `mvi` in each 5-mm depth slab, and the **background** is the bottom 50 %. `mvi` is computed by the scanner software from the contrast bolus, so the mask comes from bubble data, not from the contrast-free clip being scored. **CNR** = (mean S\* in vessels − mean S\* in background) / SD of S\* in background, and **dCNR = CNR(whitened) − CNR(nominal)**.
7. **Null.** 400 "random-covariance" whitenings per recording: the same eigenvalues as R_a but Haar-random eigenvectors (fixed seeds), scored the same way.

**Recording split.** Held out: SP03–SP10, 15 recordings (SP04 contributes only SP04-Left-2; SP04-Left-1 fails the contamination check of the first result). SP01 and SP02 were used to develop the method. SP01-Left-2 shows no contrast-free vessels and serves as the **no-blood control**: whitening must not create signal there. A subject's value is the mean over its recordings.

### Result (frozen run, 8 subjects)

| endpoint | result | pre-set rule | met |
|---|---|---|---|
| **(a) vessel CNR gain**, median subject dCNR (primary) | **+0.162**, 90 % CI +0.068 to +0.216 | median > 0 and CI lower bound > 0 | yes |
| (b) subjects with a gain | 7 of 8 | ≥ 6 of 8 | yes |
| (c) gain vs 400 random-covariance whitenings with the same eigenvalues | p = 0.0025 (= 1/401; no draw's median over subjects reached +0.162) | p ≤ 0.05 | yes |
| (d) no-blood control SP01-Left-2 | dCNR −0.071 | ≤ +0.10 | yes |
| (e) coverage | 8 subjects; 400 draws in all 15 recordings | ≥ 6 subjects; 400 draws everywhere | yes |
| noise floor change (ROI median, W\*) | **−1.08 dB**, 90 % CI −1.12 to −1.01; lower in 8/8 | reported, no bar | – |
| blood change (vessel mask, mean of B\* − W\*) | **+0.00 dB**, 90 % CI −0.51 to +0.45 | reported, no bar | – |
| whitening matrix from a **different person's** recording (SP01-Left-2, all blocks) | median dCNR **+0.164**, 90 % CI +0.085 to +0.223; 7/8 | checker's extra arm, not pre-registered | – |

Per subject (dCNR): SP03 +0.008, SP04 +0.827, SP05 +0.141, SP06 +0.225, SP07 +0.128, SP08 −0.070, SP09 +0.208, SP10 +0.183. The 90 % CIs are percentile bootstraps of the median over subjects (10 000 resamples).

**Pass rule.** PASS if (a)–(d) all hold. FAIL(WIDE) if the median is > 0 and (b)–(d) hold but the CI lower bound is ≤ 0. INCONCLUSIVE if (e) fails or the control is missing. FAIL otherwise. Nothing is ever called "confirmed".

**What it means.**
- Whitening lowers the noise floor by about 1 dB while coherent blood echo passes with unit gain. Per voxel, blood-to-noise therefore improves by about 1 dB.
- That helps the map only where it is noise-limited. In recordings whose nominal CNR is already high (SP03, SP08), the gain is about zero or slightly negative.
- A matrix taken from another person's recording works as well as the per-session one. The noise structure is fixed in the device, so one calibration matrix per probe would do.

### Caveats (these travel with the result)

- **Not an independent replication.** The 8 held-out subjects are the same ones used by the first, pre-registered whitening test (which narrowly failed: median +0.107, CI −0.010 to +0.143) and by that test's post hoc re-score on line-free bands (8/8 positive, median +0.169). The answer was therefore roughly known in advance. An independent test needs new recordings, ideally from another probe or site.
- **The 250 Hz guard is not new.** It follows from a PRF-only formula, but its value equals the "≥ 250 Hz from any line" that the first test's checker chose after seeing these held-out data.
- **The CI is fragile.** It is a percentile bootstrap of a median over n = 8. Leaving out any one of SP04, SP05, SP06, SP07, SP09 or SP10 lowers the CI lower bound to as little as +0.008 (it stays above 0 in every leave-one-out).
- **The random-covariance null is weak.** Random eigenvectors raise the noise floor (median +0.27 dB), so the null is centred below zero (median −0.078). p = 1/401 says only that the specific eigenvectors matter, not how large the gain is. Per recording, whitening beat the random draws in only 53 %, 27 % and 12.5 % of draws in SP05-Left, SP08-Left and SP09-Left-2. The cross-subject arm is the more informative control.
- **Small lever, large gap.** About 1 dB against a deficit of about 20 dB (the first result: blood is −22 dB per element per transmit).
- **One probe, one site, 8 subjects.**

### How to run

`whitening.py` needs two things per recording that the first result does not: the vendor `mvi` volume (for the vessel mask) with its grid, and the per-element transmit delays. `get_data.py --refs` saves them in `data/<recording>/refs.npz` (about 22 MB of range reads per recording). It works on recordings you have already downloaded and does not touch `clip.h5`. The 256 frames of the +10 s clip are not used here.

```bash
python get_data.py --recordings SP04-Left-2 SP06-Left SP01-Left-2 --refs   # or: --all --refs
python whitening/whitening.py                       # every held-out recording + the control found in data/
python whitening/whitening.py --recordings SP04-Left-2 SP06-Left --draws 20
python whitening/whitening.py --draws 400           # the full frozen design (verdict needs all 15 + control)
```

- It prints a per-recording table (nominal and whitened CNR, dCNR, the frozen run's dCNR and the difference, the cross-subject arm, noise floor and blood change, and the random draws), then the subject-level summary. It writes `results/whitening/whitening_results.json`, one JSON per recording and `whitening.png`.
- The cross-subject arm is computed when SP01-Left-2 is in the data folder (`--cross none` skips it).
- The pass rule is evaluated only when all 15 held-out recordings and the control are present. The verdict also needs 400 draws in every recording; with fewer it reports INCONCLUSIVE.
- `--draws N` adds the random-covariance null with the same seeds as the frozen run, so draw j here is draw j there.

### Runtime

Measured on 2026-10-24/25 on a MacBook (Apple silicon, 24 GB) with other jobs running (load 3–8), `--threads 8`:

| run | time |
|---|---|
| `get_data.py --refs` (5 recordings already present) | 86 s |
| one recording, nominal + whitened + cross-subject arm, no random draws | 42 s (plus 4 s once for the cross-subject matrix) |
| each random draw | about 12 s per recording |
| one recording with 20 draws (`--draws 20`) | 266–286 s |
| one recording with 400 draws (SP04-Left-2) | 78 min (4671 s) |
| **full frozen design on CPU** (15 recordings x 400 draws, plus the control) | **about 20 h** |

Peak memory is about 8–10 GB (`--batch 2`; lower it to 1 to save about 1 GB). The frozen 400-draw run computed the per-arm step (channel transform, delay-and-sum, band powers) on the GPU with a Metal kernel. On a development recording that kernel matched the CPU path within 1.1e-6 (relative). `whitening.py` is CPU-only (numpy/scipy).

### Our reproduction (`whitening/example_output/`)

We ran `whitening.py` on 4 held-out recordings and the control, with 20 random draws each. These came from the campaign's local copy of the same Open-H-RF chunks, repackaged into the `get_data.py` format without re-downloading; `refs.npz` was fetched with `get_data.py --refs`, and its `mvi` and `t0_delays` are bit-identical to the local copy.

| recording | CNR nominal | CNR whitened | dCNR, ours | dCNR, frozen | cross-subject arm, ours (checker) | noise floor | blood |
|---|---|---|---|---|---|---|---|
| SP04-Left-2 | 2.211 | 3.038 | +0.82695 | +0.82695 | +0.812 (+0.812) | −1.10 dB | +0.03 dB |
| SP06-Left | 0.656 | 0.994 | +0.33771 | +0.33771 | +0.317 (+0.317) | −1.10 dB | +0.51 dB |
| SP08-Left | 3.658 | 3.524 | −0.13356 | −0.13356 | −0.172 (−0.172) | −0.78 dB | +0.29 dB |
| SP09-Left-1 | 3.345 | 3.796 | +0.45102 | +0.45102 | +0.489 (+0.490) | −1.21 dB | −0.07 dB |
| SP01-Left-2 (control) | −0.000 | −0.072 | −0.07149 | −0.07149 | – | −0.84 dB | n/a (no blood excess) |

- Every per-recording dCNR, nominal CNR, noise-floor and blood change matches the frozen run within 1.3e-6 (dCNR within 5.3e-7). The first 20 random draws of each recording match the frozen run's draws 1–20 within 8e-7.
- The cross-subject arm matches the independent checker's within 2.6e-4. The checker used a different interpolation of `mvi` and a different summation order.
- With 4 of 8 subjects (and only one complete subject), these runs show that the code reproduces the frozen numbers. They are not the study result; for that, run all 15 held-out recordings.
- We also ran SP04-Left-2 with all 400 draws (`whitening/example_output/SP04-Left-2_400draws.json`). All 400 draws match the frozen run within 1.2e-6 in dCNR, and whitening beats all 400, as in the frozen run.

## Licences and attribution

- **Code** (`get_data.py`, `temple_snr.py`, `temple_snr.ipynb`, `whitening/whitening.py`): MIT License, copyright (c) 2026 MDavis3; full text in `LICENSE`.
- **Data**: Open-H-RF, Resolve Stroke contribution, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit: Aitana Waelbroeck, Carl Ferlay, Arthur Chavignon, Maxence Reberol, Vincent Hingot (Resolve Stroke). If you use it, cite the Open-H-RF paper (arXiv:2610.13616). Modification: we store only a subset of frames (the pre-bolus clip and 256 frames of the +10 s clip) and a subset of metadata (with `--refs`, also the vendor `mvi` reference volume, its grid and the transmit delays). No data are redistributed with this package; `get_data.py` fetches them from the original source. Only CC BY 4.0 data are used, and no CC BY-NC data.
