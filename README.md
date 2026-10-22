# temple-snr-repro

Reproduce one number yourself, from public data: **the signal-to-noise ratio of contrast-free blood on a single ultrasound receive element, per transmitted pulse, through the adult temporal bone at about 40 mm depth.**

**Result on all 20 recordings (10 subjects):** −22.4 dB at 37.5–42.5 mm (90 % CI −24.8 to −18.9 dB). The pre-set precision rule was **not** met at 40–45 mm (CI 6.45 dB wide, limit 6.0 dB), so the formal verdict is **FAIL** (the script prints `FAIL`; the failing criterion is P2 width, shown as `FAIL(WIDE)` on the P2 line). Details are in [Full result](#full-result-all-20-recordings).

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
6. **Noise.** N = mean spectral density in 1500 ≤ |f| ≤ 2000 Hz, which is white noise at these depths (blood would need more than about 0.57 m/s to reach it).
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

## Licences and attribution

- **Code** (`get_data.py`, `temple_snr.py`, `temple_snr.ipynb`): MIT License, copyright (c) 2026 MDavis3; full text in `LICENSE`.
- **Data**: Open-H-RF, Resolve Stroke contribution, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit: Aitana Waelbroeck, Carl Ferlay, Arthur Chavignon, Maxence Reberol, Vincent Hingot (Resolve Stroke). If you use it, cite the Open-H-RF paper (arXiv:2610.13616). Modification: we store only a subset of frames (the pre-bolus clip and 256 frames of the +10 s clip) and a subset of metadata. No data are redistributed with this package; `get_data.py` fetches them from the original source. Only CC BY 4.0 data are used, and no CC BY-NC data.
