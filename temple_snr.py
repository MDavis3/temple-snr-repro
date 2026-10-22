# SPDX-License-Identifier: MIT
"""Contrast-free blood SNR per receive element through the adult temporal window.

Input: recordings fetched by get_data.py (Open-H-RF / Resolve Stroke, CC BY 4.0).
For each recording, element and depth gate:
  1. decode the 4000-frame pre-bolus clip (I/Q int16) and undo the hardware TGC;
  2. gate: keep the fast-time samples whose round trip (diverging wave from the virtual source) maps to the gate depth;
  3. slow-time spectrum per (sample, element): mean removed, Blackman-Harris window, 4000-point FFT (1 Hz bins);
  4. notch the frame-periodic artefact lines at multiples of PRF/4 (1000, 2000, 3000 = -1000 Hz);
  5. clutter filter: the wall band |f| < 100 Hz (tissue and skull echo) is excluded;
  6. noise: flat floor of the spectrum at 1500-2000 Hz, a band blood at these depths does not reach;
  7. SNR = blood-band power above the floor / white-noise power per transmit (estimator E, primary).
Cross-check X uses a different route: time-domain period-4 subtraction, Butterworth high-pass, noise from a
4th-order temporal difference. Contamination check C: the contrast clip (+10 s) must be >= 6 dB above the
pre-bolus clip in noise-subtracted blood-band power, at gate shifts of -1, 0, +1 sample.
Aggregation: median over elements per recording; subject = linear mean of its included recordings (so with all 20
recordings each subject averages its 2); population = median over subjects, 90 % CI by a two-stage cluster bootstrap.
These definitions were fixed in writing before the 20-recording analysis was run; README.md gives them in full.

usage: python temple_snr.py [--data-dir data] [--out results] [--recordings SP01-Left-1 ...]
  By default every recording found in <data-dir>/*/clip.h5 is analysed (run get_data.py --all for the full study).
"""
import argparse, glob, json, os, time

import numpy as np
import h5py
import hdf5plugin  # noqa: F401  (Blosc filter)
from scipy import signal

# ============================================================================ CONFIG (all analysis choices)
CONFIG = dict(
    prf_hz=4000.0,
    primary_gates={"P1": (37.5, 42.5), "P2": (40.0, 45.0)},          # mm; P1 = headline (40 mm)
    depth_gates={f"G{z}": (z, z + 5.0) for z in (30, 35, 40, 45, 50, 55)},
    contamination_gate=(30.0, 60.0), contamination_min_db=6.0, contamination_shifts=(-1, 0, 1),
    clip_level=32000,            # |I| or |Q| >= this in any baseline frame: drop the (sample, element) cell
    max_dropped_cells=0.5,       # element invalid in a gate if more of its cells are dropped
    min_valid_elements=32,
    wall_hz=100.0,               # clutter filter (E: excluded band; X: Butterworth high-pass cutoff)
    blood_band_hz=(100.0, 1500.0), noise_band_hz=(1500.0, 2000.0),
    line_step_hz=1000.0,         # artefact lines at k * PRF/4
    notch_halfwidth_hz=8.0,      # baseline clip, 1 Hz bins
    control_notch_bins=4,        # control clip (256 frames, 15.625 Hz bins)
    x_order=4, x_trim_frames=200, x_diff_order=4,
    bootstrap_reps=4000, bootstrap_seed=20261022, res_floor_reps=500, res_floor_min=10 ** -3.5,
    pass_min_subjects=8, pass_max_ci_width_db=6.0, pass_max_ex_db=3.0,
)
# ==========================================================================================================


def db(x):
    return float(10 * np.log10(x)) if x is not None and np.isfinite(x) and x > 0 else None


def load_meta(path):
    with h5py.File(path, "r") as f:
        g = lambda k: np.asarray(f["meta/" + k.replace("/", "__")][()])
        s = "tracks/track_0/scan/"
        sub = g("metadata/subject/id").item()
        return dict(geom=g("probe/probe_geometry").astype(float), tgc=g(s + "tgc_gain_curve").astype(float),
                    t0=float(g(s + "initial_times").ravel()[0]), fs=float(g(s + "sampling_frequency")),
                    c=float(g(s + "sound_speed")), vsrc=-float(g(s + "focus_distances").ravel()[0]),
                    subject=(sub.decode() if isinstance(sub, bytes) else str(sub)) or str(f.attrs["recording"])[:4],
                    name=str(f.attrs["recording"]),
                    n_samples=f["clip"].shape[1])


def gate_mask(m, lo_mm, hi_mm, shift=0):
    """(n_samples, n_elements) bool: sample n of element i images depth [lo, hi) straight below that element.
    Two-way time = (virtual source -> point) - vsrc + (point -> element); shift in samples."""
    tn = m["t0"] + (np.arange(m["n_samples"]) + shift) / m["fs"]
    r = np.hypot(m["geom"][:, 0], m["geom"][:, 1]); vs = m["vsrc"]
    t = lambda z: (np.sqrt(r ** 2 + (z * 1e-3 + vs) ** 2) - vs + z * 1e-3) / m["c"]
    return (tn[:, None] >= t(lo_mm)[None]) & (tn[:, None] < t(hi_mm)[None])


def band_masks(n, cfg, notch_hz):
    f = np.fft.fftfreq(n, 1 / cfg["prf_hz"]); af = np.abs(f)
    P = cfg["prf_hz"]; wrap = lambda d: (d + P / 2) % P - P / 2
    notch = np.any([np.abs(wrap(f - k * cfg["line_step_hz"])) <= notch_hz for k in (1, 2, 3)], 0)
    B = (af >= cfg["blood_band_hz"][0]) & (af < cfg["blood_band_hz"][1])
    W = (af >= cfg["noise_band_hz"][0]) & (af <= cfg["noise_band_hz"][1])
    return f, notch, B, W


def spectra(x):
    """Periodogram along axis 0 (mean removed, 4-term Blackman-Harris). Normalisation cancels in all ratios."""
    w = signal.windows.blackmanharris(x.shape[0], sym=False)
    return np.abs(np.fft.fft((x - x.mean(0)) * w[:, None, None], axis=0)) ** 2 / np.sum(w ** 2)


def x_constants(cfg):
    """White-noise pass fractions for cross-check X: kappa = mean |H|^4 (zero-phase HPF), g_d = mean |H|^4 |2 sin(pi f/PRF)|^(2 d).
    Computed from the filter response, not simulated: a 4000 x 256 white-noise simulation carries ~0.2 % sampling
    error, while at -22 dB the blood power is only ~0.6 % of the total noise power, so the simulated constants can
    shift X by > 1 dB (+1.5 dB with one fixed seed; the sign is random)."""
    sos = signal.butter(cfg["x_order"], cfg["wall_hz"], btype="highpass", fs=cfg["prf_hz"], output="sos")
    f = (np.arange(1 << 18) + 0.5) / (1 << 18) * cfg["prf_hz"] - cfg["prf_hz"] / 2
    H4 = np.abs(signal.sosfreqz(sos, worN=np.abs(f), fs=cfg["prf_hz"])[1]) ** 4
    D = np.abs(2 * np.sin(np.pi * f / cfg["prf_hz"])) ** (2 * cfg["x_diff_order"])
    return sos, float(H4.mean()), float((H4 * D).mean())


def cell_stats(path, m, rows, spec_mask, cfg):
    """Per (sample, element) band means of the slow-time spectrum (baseline and control) and X powers."""
    r0, r1 = rows
    with h5py.File(path, "r") as f:
        raw = f["clip"][:, r0:r1]; ctrl = f["control"][:, r0:r1]       # (frames, samples, elements, I/Q) int16
    nF, nR, nE = raw.shape[:3]; L = cfg["clip_level"]
    clipped = np.zeros((nR, nE), bool)
    for a in range(0, nF, 500):
        blk = raw[a:a + 500]; clipped |= ((blk >= L) | (blk <= -L)).any(-1).any(0)
    clipped_ctrl = clipped | ((ctrl >= L) | (ctrl <= -L)).any(-1).any(0)
    _, nt, B, W = band_masks(nF, cfg, cfg["notch_halfwidth_hz"])
    _, ntc, Bc, Wc = band_masks(ctrl.shape[0], cfg, cfg["control_notch_bins"] * cfg["prf_hz"] / ctrl.shape[0] + 1e-9)
    sos, kap, g_d = x_constants(cfg); T = cfg["x_trim_frames"]
    keys = ["D", "N", "D_nonotch", "N_nonotch", "Dc", "Nc", "Px", "Dx"]
    out = {k: np.zeros((nR, nE)) for k in keys}; spec = np.zeros(nF)
    for a in range(0, nR, 8):                                            # 8 fast-time samples at a time
        b = min(nR, a + 8); tg = m["tgc"][r0 + a:r0 + b][None, :, None]
        x = (raw[:, a:b, :, 0] + 1j * raw[:, a:b, :, 1]) / tg
        S = spectra(x)
        out["D"][a:b] = S[B & ~nt].mean(0); out["N"][a:b] = S[W & ~nt].mean(0)
        out["D_nonotch"][a:b] = S[B].mean(0); out["N_nonotch"][a:b] = S[W].mean(0)
        sm = spec_mask[a:b] & ~clipped[a:b]                              # mean noise-normalised spectrum (figure)
        if sm.any():
            spec += (S / out["N"][a:b][None])[:, sm].sum(1)
        xc = (ctrl[:, a:b, :, 0] + 1j * ctrl[:, a:b, :, 1]) / tg
        Sc = spectra(xc); out["Dc"][a:b] = Sc[Bc & ~ntc].mean(0); out["Nc"][a:b] = Sc[Wc & ~ntc].mean(0)
        # cross-check X: exact period-4 subtraction, Butterworth HPF, 4th-difference noise
        xp = x.copy()
        for k in range(4):
            xp[k::4] -= x[k::4].mean(0)
        xp += x.mean(0)
        y = signal.sosfiltfilt(sos, xp, axis=0)[T:nF - T]
        out["Px"][a:b] = np.mean(np.abs(y) ** 2, 0)
        out["Dx"][a:b] = np.mean(np.abs(np.diff(y, cfg["x_diff_order"], axis=0)) ** 2, 0) / g_d
    out.update(clipped=clipped, clipped_ctrl=clipped_ctrl, kappa=kap, spec=spec / max((spec_mask & ~clipped).sum(), 1))
    return out


def element_snr(cs, mask, cfg, D="D", N="N"):
    """Per-element linear SNR in a gate mask (rows already cropped); NaN for invalid elements."""
    keep = mask & ~cs["clipped"]; n_all = mask.sum(0); n = keep.sum(0)
    valid = (n_all > 0) & (n >= (1 - cfg["max_dropped_cells"]) * n_all) & (n > 0)   # validity from the baseline clip
    if D == "Dc":                                          # control clip: also drop cells clipped in the control frames
        keep = mask & ~cs["clipped_ctrl"]; n = keep.sum(0); valid &= n > 0
    avg = lambda A: np.where(valid, np.where(keep, A, 0).sum(0) / np.maximum(n, 1), np.nan)
    if D == "Px":                                                       # X: (P - kappa sigma^2) / sigma^2
        s2 = avg(cs["Dx"]); return (avg(cs["Px"]) - cs["kappa"] * s2) / s2
    d, nn = avg(cs[D]), avg(cs[N])
    bw = cfg["blood_band_hz"][1] - cfg["blood_band_hz"][0]
    return (d - nn) * 2 * bw / (nn * cfg["prf_hz"])                      # band power / noise power per transmit


def column_ids(geom):
    return np.unique(np.round(geom[:, 0], 6), return_inverse=True)[1]   # 32 columns of 8 elements


def col_resample_median(e, cols, rng):
    """Median of e after resampling whole element columns (8 adjacent elements) with replacement."""
    nc = cols.max() + 1; order = np.argsort(cols, kind="stable"); per = len(cols) // nc
    pick = rng.integers(0, nc, nc)
    return np.nanmedian(e[order].reshape(nc, per)[pick])


def analyse_recording(path, cfg=CONFIG):
    m = load_meta(path); cols = column_ids(m["geom"])
    gates = {**cfg["primary_gates"], **cfg["depth_gates"]}
    lo = min([g[0] for g in gates.values()] + [cfg["contamination_gate"][0]])
    hi = max([g[1] for g in gates.values()] + [cfg["contamination_gate"][1]])
    M = gate_mask(m, lo, hi, -2) | gate_mask(m, lo, hi, 2); rr = np.where(M.any(1))[0]; rows = (rr[0], rr[-1] + 1)
    crop = lambda mk: mk[rows[0]:rows[1]]
    cs = cell_stats(path, m, rows, crop(gate_mask(m, *cfg["primary_gates"]["P1"])), cfg)
    res = dict(recording=m["name"], subject=m["subject"], gates={}, per_element={})
    for g, (a, b) in gates.items():
        mk = crop(gate_mask(m, a, b))
        e, ex, e0 = element_snr(cs, mk, cfg), element_snr(cs, mk, cfg, "Px"), element_snr(cs, mk, cfg, "D_nonotch", "N_nonotch")
        nv = int(np.isfinite(e).sum())
        res["per_element"][g] = e
        res["gates"][g] = dict(n_valid=nv, clip_ok=nv >= cfg["min_valid_elements"] and nv > 0.5 * len(e),
                               E_db=db(np.nanmedian(e)), X_db=db(np.nanmedian(ex)), E_nonotch_db=db(np.nanmedian(e0)),
                               E_lin=float(np.nanmedian(e)), X_lin=float(np.nanmedian(ex)))
        res["per_element"][g + "_X"] = ex
    # contamination check C (gate 30-60 mm, shifts -1/0/+1 sample)
    C = {}
    for s in cfg["contamination_shifts"]:
        rng = np.random.default_rng(cfg["bootstrap_seed"])
        mk = crop(gate_mask(m, *cfg["contamination_gate"], shift=s))
        eb, ec = element_snr(cs, mk, cfg), element_snr(cs, mk, cfg, "Dc", "Nc")
        sb, sc = np.nanmedian(eb), np.nanmedian(ec)
        se = np.std([col_resample_median(eb, cols, rng) for _ in range(cfg["res_floor_reps"])], ddof=1)
        sb_eff = max(sb, 2 * se, cfg["res_floor_min"])
        C[s] = float(10 * np.log10(sc / sb_eff)) if sc > 0 else -np.inf
    res["contamination_db"] = C; res["contamination_min_db"] = min(C.values())
    res["passes_contamination"] = res["contamination_min_db"] >= cfg["contamination_min_db"]
    res["spectrum_P1"] = dict(f_hz=np.fft.fftshift(np.fft.fftfreq(len(cs["spec"]), 1 / cfg["prf_hz"])),
                              power_over_noise=np.fft.fftshift(cs["spec"]))
    res["columns"] = cols
    return res


def population(results, gate, cfg=CONFIG, key="E", boot=True):
    """Median over included subjects of the subject value (linear mean of its included recordings' element medians);
    90 % CI by two-stage cluster bootstrap (subjects, then element columns within each recording)."""
    subj = {}
    for r in results:
        if r["passes_contamination"] and r["gates"][gate]["clip_ok"]:
            subj.setdefault(r["subject"], []).append(r)
    if not subj:
        return dict(gate=gate, key=key, n_subjects=0)
    pe = lambda r: r["per_element"][gate if key == "E" else gate + "_X"]
    q = [np.mean([np.nanmedian(pe(r)) for r in rs]) for rs in subj.values()]
    rng = np.random.default_rng(cfg["bootstrap_seed"]); S = list(subj.values()); reps = []
    if not boot:
        return dict(gate=gate, key=key, n_subjects=len(S), median_db=db(np.median(q)))
    for _ in range(cfg["bootstrap_reps"]):
        reps.append(np.median([np.mean([col_resample_median(pe(r), r["columns"], rng) for r in S[i]])
                               for i in rng.integers(0, len(S), len(S))]))
    lo, hi = np.percentile(reps, [5, 95])
    return dict(gate=gate, key=key, n_subjects=len(S), median_db=db(np.median(q)), ci90_db=[db(lo), db(hi)],
                ci_width_db=float(10 * np.log10(hi / lo)) if lo > 0 else float("inf"),
                subject_db={k: db(v) for k, v in zip(subj, q)}, median_lin=float(np.median(q)),
                subject_recordings={k: [r["recording"] for r in rs] for k, rs in subj.items()})


def verdict(pE, pX, cfg=CONFIG):
    lab = []
    if pE["n_subjects"] < cfg["pass_min_subjects"]: lab.append("FEW")
    if not pE["n_subjects"] or pE["ci_width_db"] >= cfg["pass_max_ci_width_db"]: lab.append("WIDE")
    if not pE["n_subjects"] or pE["median_db"] is None or pX["median_db"] is None or \
            abs(pE["median_db"] - pX["median_db"]) > cfg["pass_max_ex_db"]: lab.append("DISAGREE")
    return "PASS" if not lab else "FAIL(" + "+".join(lab) + ")"


def overall_verdict(pops, cfg=CONFIG):
    """Protocol rule: PASS only if every primary gate (P1 and P2) passes."""
    v = [verdict(pops[(g, "E")], pops[(g, "X")], cfg) for g in cfg["primary_gates"]]
    return "PASS" if all(x == "PASS" for x in v) else "FAIL"


def f1(v, w=7):
    if v is None: return f"{'<res':>{w}}"
    if isinstance(v, float) and np.isinf(v): return f"{'-inf':>{w}}"
    return f"{v:{w}.1f}"


def print_table(results, pops, cfg=CONFIG):
    dg = list(cfg["depth_gates"])
    print("\nContrast-free blood SNR, dB per element per transmit (median over valid elements)")
    print(f"{'recording':<13}{'Cmin':>6}{'incl':>5}{'nP1':>5}{'P1 E':>7}{'P1 X':>7}{'P1 no-notch':>12}{'P2 E':>7}{'P2 X':>7}  |"
          + "".join(f"{g:>6}" for g in dg))
    for r in results:
        G = r["gates"]
        print(f"{r['recording']:<13}{f1(r['contamination_min_db'], 6)}{'yes' if r['passes_contamination'] else 'no':>5}"
              f"{G['P1']['n_valid']:>5}{f1(G['P1']['E_db'])}{f1(G['P1']['X_db'])}{f1(G['P1']['E_nonotch_db'], 12)}"
              f"{f1(G['P2']['E_db'])}{f1(G['P2']['X_db'])}  |" + "".join(f1(G[g]['E_db'], 6) for g in dg))
    print(f"Cmin = contamination check, min over gate shifts of contrast/pre-bolus blood-band power, 30-60 mm (dB);"
          f" included if >= {cfg['contamination_min_db']}. '<res' = not above the noise floor.")
    pg = list(cfg["primary_gates"])
    subs = sorted({s for g in pg for s in pops[(g, "E")].get("subject_db", {})})
    if subs:
        print("\nPer subject (linear mean of its included recordings, then dB)")
        print(f"{'subject':<9}{'recordings used':<28}" + "".join(f"{g + ' ' + k:>7}" for g in pg for k in "EX"))
        for sb in subs:
            recs = pops[(pg[0], "E")].get("subject_recordings", {}).get(sb) or pops[(pg[-1], "E")]["subject_recordings"][sb]
            print(f"{sb:<9}{' '.join(recs):<28}" + "".join(
                f1(pops[(g, k)].get("subject_db", {}).get(sb)) if sb in pops[(g, k)].get("subject_db", {}) else f"{'-':>7}"
                for g in pg for k in "EX"))
    print()
    for g in cfg["primary_gates"]:
        pE, pX = pops[(g, "E")], pops[(g, "X")]
        if not pE["n_subjects"]:
            print(f"{g}: no subject included"); continue
        print(f"{g} {cfg['primary_gates'][g]} mm: E median {f1(pE['median_db'], 0)} dB, 90% CI [{f1(pE['ci90_db'][0], 0)}, "
              f"{f1(pE['ci90_db'][1], 0)}] (width {pE['ci_width_db']:.1f} dB); X median {f1(pX['median_db'], 0)} dB; "
              f"n = {pE['n_subjects']} subjects; protocol verdict {verdict(pE, pX, cfg)}")
    print(f"Overall verdict (P1 and P2 must both PASS: >= {cfg['pass_min_subjects']} subjects, CI width < "
          f"{cfg['pass_max_ci_width_db']} dB, |E - X| <= {cfg['pass_max_ex_db']} dB): {overall_verdict(pops, cfg)}")


def make_figure(results, pops, path, cfg=CONFIG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8))
    dg = cfg["depth_gates"]; z = [np.mean(v) for v in dg.values()]
    cmap = plt.get_cmap("tab20")
    colors = {r["recording"]: cmap(i % 20) for i, r in enumerate(results)}   # distinct colour per recording (up to 20)
    for r in results:
        y = [r["gates"][g]["E_db"] if r["gates"][g]["E_db"] is not None else np.nan for g in dg]
        inc = r["passes_contamination"]
        ax[0].plot(z, y, "-o" if inc else ":x", ms=4, lw=1.2 if inc else 0.8, color=colors[r["recording"]] if inc else "0.6",
                   label=r["recording"] + ("" if inc else " (excluded: C)"))
    med = [population(results, g, cfg, boot=False).get("median_db") for g in dg]
    if any(v is not None for v in med):
        ax[0].plot(z, [v if v is not None else np.nan for v in med], "k-", lw=2.5, label="median, included subjects")
    ax[0].set(xlabel="gate centre depth (mm)", ylabel="blood SNR per element per transmit (dB)",
              title="Contrast-free blood SNR vs depth (estimator E)")
    ax[0].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0); ax[0].grid(alpha=0.3)
    k = np.ones(9) / 9
    for r in results:                           # 9-bin moving average; reflect padding avoids a false dip at the edges
        s = r["spectrum_P1"]; sm = np.convolve(np.pad(s["power_over_noise"], 4, mode="reflect"), k, "valid")
        ax[1].plot(s["f_hz"], 10 * np.log10(sm), lw=0.7, color=colors[r["recording"]])   # recordings labelled in the left legend
    for sg in (-1, 1):
        ax[1].axvspan(sg * cfg["blood_band_hz"][0], sg * cfg["blood_band_hz"][1], color="tab:red", alpha=0.07,
                      label="blood band" if sg > 0 else None)
        ax[1].axvspan(sg * cfg["noise_band_hz"][0], sg * cfg["noise_band_hz"][1], color="tab:blue", alpha=0.10,
                      label="noise band" if sg > 0 else None)
    ax[1].axvspan(-cfg["wall_hz"], cfg["wall_hz"], color="0.5", alpha=0.15, label="clutter (wall) band, excluded")
    ax[1].set(xlabel="slow-time (Doppler) frequency (Hz)", ylabel="power / noise floor (dB, 9-bin smoothed)", ylim=(-0.5, 3),
              title="Mean spectrum, P1 gate 37.5-42.5 mm (shown without the notch)\n"
                    "spikes at +-1000 and -2000 Hz are scanner lines, notched (+-8 Hz) before SNR")
    ax[1].title.set_fontsize(9); ax[1].legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def run(data_dir="data", out_dir="results", cfg=CONFIG, verbose=True, recordings=None):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(data_dir, "*", "clip.h5")))
    if recordings:
        have = {os.path.basename(os.path.dirname(p)): p for p in files}
        missing = [n for n in recordings if n not in have]
        if missing:
            raise SystemExit(f"not downloaded in {data_dir}: {missing} - run get_data.py --recordings {' '.join(missing)}")
        files = [have[n] for n in sorted(recordings)]
    if not files:
        raise SystemExit(f"no recordings in {data_dir}/*/clip.h5 - run get_data.py first")
    if verbose:
        print(f"analysing {len(files)} recording(s) from {data_dir}", flush=True)
    results = []
    for p in files:
        t0 = time.time(); r = analyse_recording(p, cfg); results.append(r)
        if verbose:
            print(f"{r['recording']}: P1 E {f1(r['gates']['P1']['E_db'], 0)} dB, C_min {f1(r['contamination_min_db'], 0)} dB "
                  f"({time.time() - t0:.0f} s)", flush=True)
    pops = {(g, k): population(results, g, cfg, k) for g in cfg["primary_gates"] for k in ("E", "X")}
    print_table(results, pops, cfg)
    make_figure(results, pops, os.path.join(out_dir, "temple_snr.png"), cfg)
    skip = ("per_element", "spectrum_P1", "columns")
    js = dict(config=cfg, overall_verdict=overall_verdict(pops, cfg), population={f"{g}_{k}": dict(v, verdict=verdict(pops[(g, 'E')], pops[(g, 'X')], cfg) if k == "E" else None)
                                      for (g, k), v in pops.items()},
              recordings=[{k: v for k, v in r.items() if k not in skip} for r in results])
    json.dump(js, open(os.path.join(out_dir, "temple_snr_results.json"), "w"), indent=1, default=str)
    np.savez(os.path.join(out_dir, "per_element_snr_linear.npz"),
             **{f"{r['recording']}__{g}": v for r in results for g, v in r["per_element"].items()})
    return results, pops


def selftest(tmp_dir, snr_db=-22.0, cfg=CONFIG):
    """Synthetic recording: white noise + 150-600 Hz band 'blood' at snr_db + 1000/2000 Hz lines (+7 dB per bin)
    + strong static clutter. E and X must recover snr_db within 0.5 dB."""
    rng = np.random.default_rng(1); nF, nS, nc = 4000, 100, 32
    geom = np.array([[(c - 15.5) * 0.5e-3, (k - 3.5) * 2e-3, 0] for c in range(nc) for k in range(8)])
    f = np.fft.fftfreq(nF, 1 / cfg["prf_hz"]); band = (np.abs(f) >= 150) & (np.abs(f) <= 600)
    sig, sd = 200.0, 10 ** (snr_db / 20)
    tline = np.exp(2j * np.pi * 1000 * np.arange(nF) / cfg["prf_hz"]); tline2 = np.cos(np.pi * np.arange(nF))
    line_amp = sig * np.sqrt(10 ** 0.7 / nF)
    def clip(boost):
        x = np.empty((nF, nS, len(geom), 2), np.int16)
        for a in range(0, nS, 10):
            sh = (nF, 10, len(geom))
            n = sig * (rng.standard_normal(sh) + 1j * rng.standard_normal(sh)) / np.sqrt(2)
            w = np.fft.fft(rng.standard_normal(sh) + 1j * rng.standard_normal(sh), axis=0) * band[:, None, None]
            b = np.fft.ifft(w, axis=0); b *= sig * sd * boost / np.sqrt(np.mean(np.abs(b) ** 2))
            y = n + b + 30 * sig + line_amp * (tline + tline2)[:, None, None]
            x[:, a:a + 10, :, 0], x[:, a:a + 10, :, 1] = np.round(y.real), np.round(y.imag)
        return x
    d = os.path.join(tmp_dir, "SYNTH"); os.makedirs(d, exist_ok=True); p = os.path.join(d, "clip.h5")
    with h5py.File(p, "w") as h:
        h["clip"] = clip(1.0); h["control"] = clip(10.0)[:256]
        s = "meta/tracks__track_0__scan__"
        h["meta/probe__probe_geometry"] = geom; h[s + "tgc_gain_curve"] = np.ones(nS)
        h[s + "initial_times"] = [38.0e-6]; h[s + "sampling_frequency"] = 2.03125e6; h[s + "sound_speed"] = 1540.0
        h[s + "focus_distances"] = [-0.022]; h["meta/metadata__subject__id"] = "SYN"; h.attrs["recording"] = "SYNTH"
    # pass/fail on a 30-mm gate (~80 samples per element); a 5-mm gate on ONE recording has ~0.5 dB sampling SD
    tcfg = dict(cfg, primary_gates={"S30-60": (30.0, 60.0), **cfg["primary_gates"]})
    r = analyse_recording(p, tcfg); ok = True
    for g in tcfg["primary_gates"]:
        for k in ("E_db", "X_db"):
            v = r["gates"][g][k]; good = v is not None and abs(v - snr_db) <= 0.5
            if g == "S30-60":
                ok &= good
            print(f"selftest {g:>7} {k[0]}: {f1(v, 0)} dB (target {snr_db})" + (f" {'ok' if good else 'FAIL'}" if g == "S30-60" else " (info, not tested)"))
    print("selftest note: only the 30-60 mm gate is tested. 5-mm gates (P1, P2) on ONE synthetic recording scatter by\n"
          "  about +-0.5-1 dB from sampling noise alone, so a P1/P2 value up to ~1 dB off target is expected, not a failure.")
    print("selftest C_min", f1(r["contamination_min_db"], 0), "dB; overall", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data"); ap.add_argument("--out", default="results")
    ap.add_argument("--selftest", action="store_true", help="run on a synthetic recording with known SNR and exit")
    ap.add_argument("--recordings", nargs="*", help="analyse only these downloaded recordings (default: all in --data-dir)")
    a = ap.parse_args(); t = time.time()
    if a.selftest:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            raise SystemExit(0 if selftest(td) else 1)
    run(a.data_dir, a.out, recordings=a.recordings)
    print(f"analysis time {time.time() - t:.0f} s")
