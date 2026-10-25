# SPDX-License-Identifier: MIT
"""Receive-noise whitening of contrast-free power Doppler (Open-H-RF / Resolve Stroke, pre-bolus clip).

The 256 receive channels of this probe share correlated noise (32 groups of 8 channels, one shared noise source per
group). Whitening that noise before delay-and-sum (DAS) beamforming lowers the image noise floor by about 1 dB, while
coherent blood echo passes unchanged. This script measures that, per recording:

  1. Bands, from PRF only (no data): artefact lines at k*PRF/4 (k = 0..3; k = 0 is tissue clutter at DC), guard PRF/16
     around every line. Blood band B* = 250-750 Hz, noise-reference band W* = 1250-1750 Hz (both signs of f).
  2. Frames 0-3999 of the pre-bolus clip, 8 blocks of 500. Per block: slow-time mean removal, Blackman-Harris-4 window,
     slow-time FFT, keep the B* and W* bins.
  3. Noise covariance per recording: W* bins, fast-time samples 50-299, each sample normalised by its mean power.
     Cross-fitted in time: even blocks are whitened with the covariance of the odd blocks and vice versa.
  4. Whitening ("full05"): R = noise correlation matrix, R_a = 0.95 R + 0.05 I, T = D^1/2 R_a^-1/2 D^-1/2
     (D = channel noise powers, so each channel keeps its noise power), scaled so mean(diag T) = 1.
     "nominal" is T = I.
  5. TGC removed, x4 FFT upsampling in fast time, x <- T x, 3-D DAS (diverging wave from the virtual source, c = 1540 m/s,
     uniform apodization) on x, y = +/-18 mm (1 mm), z = 30-60 mm (0.5 mm). Per voxel: band-mean power D_B*, D_W*.
     All steps are linear and act on different axes, so doing the slow-time FFT first gives the same numbers as
     beamforming every frame first.
  6. S* = <D_B*> / <D_W*> - 1 per voxel (mean over 8 blocks; 0 = noise only), Gaussian smoothing sigma = 1 voxel.
     Vessel mask = top 10 % of the vendor reference volume `mvi` per 5-mm depth slab, background = bottom 50 %.
     CNR = (mean S*_vessel - mean S*_background) / std S*_background.  dCNR = CNR(full05) - CNR(nominal).
  Also: noise floor change (ROI median of 10 log10 W*(full05)/W*(nominal)), blood change (vessel mask,
  10 log10 of the mean blood excess D_B* - D_W*, full05 / nominal), a cross-subject arm (the full05 matrix built from
  another person's recording, SP01-Left-2, all 8 blocks) and optionally the matched-spectrum random-covariance null
  (same eigenvalues of R_a, Haar-random eigenvectors; seeds as in the frozen run).

These definitions, the recording split and the pass rule were frozen before the confirmatory run (README.md,
"Second result"). This file is a standalone re-implementation for reproduction; it is not the frozen code.

usage: python whitening/whitening.py [--data-dir data] [--out results/whitening] [--recordings SP06-Left ...]
                                     [--draws 0] [--cross SP01-Left-2 | --cross none] [--threads 8]
Needs get_data.py --refs for every recording analysed (vendor mvi volume and t0_delays, ~22 MB of range reads each).
"""
import argparse, json, os, time, zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import h5py
import hdf5plugin  # noqa: F401  (Blosc filter)
from scipy import signal, sparse, ndimage

# ============================================================================ fixed definitions (frozen)
NB, NBLK, U = 500, 8, 4                         # block length (frames), blocks, fast-time upsampling
S0, S1 = 50, 300                                # fast-time samples used for the noise covariance
ALPHA = 0.05                                    # shrinkage of the correlation matrix
ROI_X = np.arange(-18, 18.01, 1.0) * 1e-3       # x and y (m)
ROI_Z = np.arange(30, 60.01, 0.5) * 1e-3        # z (m)
TOP, BOTTOM, SLAB = 0.10, 0.50, 5e-3            # vessel / background quantiles per depth slab
SEED = 20261024                                 # random draws: default_rng([SEED, crc32(rec) % 100000, j, parity])
HELD_OUT = {"SP03": ["SP03-Left", "SP03-Right"], "SP04": ["SP04-Left-2"], "SP05": ["SP05-Left", "SP05-Right"],
            "SP06": ["SP06-Left", "SP06-Right"], "SP07": ["SP07-Left", "SP07-Right"], "SP08": ["SP08-Left", "SP08-Right"],
            "SP09": ["SP09-Left-1", "SP09-Left-2"], "SP10": ["SP10-Left", "SP10-Right"]}
CONTROL = "SP01-Left-2"                         # no contrast-free blood visible: whitening must not create signal
DEV = ["SP01-Left-1", "SP02-Left-1", "SP02-Left-2"]   # used to choose the method; descriptive only
EXCLUDED = {"SP04-Left-1": "fails the contamination check (C = 3.7 dB < 6 dB)"}
PASS_RULE = dict(n_subjects_min=6, n_positive_min=6, p_random_max=0.05, control_dCNR_max=0.10, draws_required=400,
                 boot_reps=10000, boot_seed=20261024)
# Frozen run (400 random draws; GPU kernel matched to the CPU path within 1.1e-6), and the independent checker's
# cross-subject arm. Used only to print a comparison column.
FROZEN = {
    "SP01-Left-1": dict(cnr_nominal=2.517634, dCNR=+0.293443, noise_floor_dB=-0.9172, dCNR_cross=None),
    "SP01-Left-2": dict(cnr_nominal=-0.000104, dCNR=-0.071486, noise_floor_dB=-0.8369, dCNR_cross=None),
    "SP02-Left-1": dict(cnr_nominal=0.268074, dCNR=+0.382584, noise_floor_dB=-1.1299, dCNR_cross=None),
    "SP02-Left-2": dict(cnr_nominal=0.020546, dCNR=+0.065566, noise_floor_dB=-1.2033, dCNR_cross=None),
    "SP03-Left": dict(cnr_nominal=5.561072, dCNR=-0.032252, noise_floor_dB=-1.0587, dCNR_cross=+0.024701),
    "SP03-Right": dict(cnr_nominal=4.795211, dCNR=+0.048923, noise_floor_dB=-0.8975, dCNR_cross=+0.059400),
    "SP04-Left-2": dict(cnr_nominal=2.210920, dCNR=+0.826954, noise_floor_dB=-1.1029, dCNR_cross=+0.811680),
    "SP05-Left": dict(cnr_nominal=0.125587, dCNR=-0.006798, noise_floor_dB=-1.1686, dCNR_cross=-0.002214),
    "SP05-Right": dict(cnr_nominal=1.002653, dCNR=+0.289109, noise_floor_dB=-1.1387, dCNR_cross=+0.280862),
    "SP06-Left": dict(cnr_nominal=0.655850, dCNR=+0.337710, noise_floor_dB=-1.1006, dCNR_cross=+0.317466),
    "SP06-Right": dict(cnr_nominal=0.378248, dCNR=+0.111606, noise_floor_dB=-1.0026, dCNR_cross=+0.115407),
    "SP07-Left": dict(cnr_nominal=1.494179, dCNR=+0.183069, noise_floor_dB=-1.2690, dCNR_cross=+0.190678),
    "SP07-Right": dict(cnr_nominal=0.251446, dCNR=+0.072221, noise_floor_dB=-0.9957, dCNR_cross=+0.065626),
    "SP08-Left": dict(cnr_nominal=3.657747, dCNR=-0.133558, noise_floor_dB=-0.7770, dCNR_cross=-0.172022),
    "SP08-Right": dict(cnr_nominal=6.132378, dCNR=-0.006824, noise_floor_dB=-0.6528, dCNR_cross=-0.056045),
    "SP09-Left-1": dict(cnr_nominal=3.345334, dCNR=+0.451024, noise_floor_dB=-1.2124, dCNR_cross=+0.489584),
    "SP09-Left-2": dict(cnr_nominal=0.183708, dCNR=-0.035084, noise_floor_dB=-0.9845, dCNR_cross=-0.030107),
    "SP10-Left": dict(cnr_nominal=0.409105, dCNR=+0.043444, noise_floor_dB=-1.0860, dCNR_cross=+0.045156),
    "SP10-Right": dict(cnr_nominal=1.262166, dCNR=+0.321983, noise_floor_dB=-0.9890, dCNR_cross=+0.331368),
}
# ==========================================================================================================
T0 = time.time()
WIN = signal.windows.blackmanharris(NB, sym=False)


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ----------------------------------------------------------------------------- data (get_data.py format)
def load_meta(data_dir, rec):
    """Acquisition parameters from <rec>/clip.h5 (meta group) and <rec>/refs.npz (get_data.py --refs)."""
    path = os.path.join(data_dir, rec, "clip.h5"); rp = os.path.join(data_dir, rec, "refs.npz")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: run  python get_data.py --recordings {rec} --refs")
    if not os.path.exists(rp):
        raise SystemExit(f"{rp} not found: run  python get_data.py --recordings {rec} --refs  (no frames are re-downloaded)")
    s = "tracks__track_0__scan__"
    with h5py.File(path, "r") as f:
        g = lambda k: np.asarray(f["meta/" + k][()])
        frames = np.asarray(f["clip"].attrs["source_frames"])
        m = dict(path=path, elem=g("probe__probe_geometry").astype(np.float64), fs=float(g(s + "sampling_frequency")),
                 fd=float(g(s + "demodulation_frequency")), c=float(g(s + "sound_speed")),
                 tinit=float(g(s + "initial_times").ravel()[0]), apex=-float(g(s + "focus_distances").ravel()[0]),
                 tgc=g(s + "tgc_gain_curve").astype(np.float64),
                 prf=1.0 / float(np.median(g(s + "time_to_next_transmit")[frames[:NB * NBLK]].ravel())),
                 shape=f["clip"].shape)
    assert m["shape"][0] >= NB * NBLK and m["shape"][1:] == (320, 256, 2), m["shape"]
    r = np.load(rp)
    m.update(mvi=r["mvi"].astype(np.float64), mvi_axes=(r["mvi_z"], r["mvi_y"], r["mvi_x"]), t0=r["t0_delays"])
    src = np.array([0.0, 0.0, -m["apex"]])
    m["d0"] = float(np.linalg.norm(m["elem"][np.argmin(m["t0"])] - src))   # the wave passes the t0 = 0 element at t = 0
    return m


def read_block(m, b):
    with h5py.File(m["path"], "r") as f:
        r = f["clip"][b * NB:(b + 1) * NB]
    return r[..., 0].astype(np.float32) + 1j * r[..., 1].astype(np.float32)        # (500, 320, 256) complex64


# ----------------------------------------------------------------------------- bands, spectra, transforms
def bands(prf):
    """B* and W* bin indices of np.fft.fftfreq(NB, 1/prf): lines at k*PRF/4, guard PRF/16 (data-independent)."""
    f = np.fft.fftfreq(NB, 1 / prf); af = np.abs(f)
    lines = np.array([0, prf / 4, prf / 2, 3 * prf / 4])
    dist = np.min(np.abs((f[:, None] - lines[None] + prf / 2) % prf - prf / 2), 1)
    free = dist >= prf / 16 - 1e-6
    return np.nonzero(free & (af <= prf / 4))[0], np.nonzero(free & (af > prf / 4) & (af <= prf / 2))[0], f


def slowfft(x, bins):
    """(500, 320, 256) block -> (len(bins), 320, 256): mean removed, BH4 window, slow-time FFT at the chosen bins."""
    x = x - x.mean(0, keepdims=True)
    out = np.empty((len(bins),) + x.shape[1:], np.complex64)
    for c0 in range(0, x.shape[2], 32):                                   # channel chunks bound the memory
        out[:, :, c0:c0 + 32] = np.fft.fft(x[:, :, c0:c0 + 32] * WIN[:, None, None], axis=0)[bins]
    return out


def noise_cov(FW):
    """W* bins (nW, 320, 256) -> (256, 256) noise covariance of samples 50-299, each sample power-normalised."""
    G = FW[:, S0:S1, :].astype(np.complex128)
    G = G / np.sqrt(np.mean(np.abs(G) ** 2, axis=(0, 2), keepdims=True))
    Z = G.reshape(-1, G.shape[2])
    return Z.conj().T @ Z / Z.shape[0]


def haar(n, rng):
    """Haar-random unitary (QR of a complex Gaussian matrix with the phase convention fixed)."""
    Z = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    return Q * (np.diag(R) / np.abs(np.diag(R)))[None, :]


def whitening(S, rng=None):
    """full05 transform from covariance S; with rng: the matched-spectrum random control (same eigenvalues of R_a,
    Haar-random eigenvectors). Returns T (256, 256) complex64 acting as x <- T x, mean(diag T) = 1."""
    d = np.real(np.diag(S)).copy(); R = S / np.sqrt(np.outer(d, d))
    Ra = (1 - ALPHA) * R + ALPHA * np.eye(len(d))
    if rng is not None:
        w = np.linalg.eigvalsh(Ra); Q = haar(len(d), rng); Ra = (Q * w) @ Q.conj().T
    w, V = np.linalg.eigh(Ra); w = np.clip(w, 1e-12 * w.max(), None)
    M = (V * w ** -0.5) @ V.conj().T
    T = (np.sqrt(d)[:, None] * M / np.sqrt(d)[None, :]).astype(np.complex64)
    return (T / np.mean(np.real(np.diag(T)))).astype(np.complex64)


def upsample(F, tgc):
    """(nb, 320, 256) -> (nb, 1280, 256): TGC removed, x4 band-limited FFT zero-padding (Nyquist bin split in two)."""
    F = F / tgc[None, :, None]
    n = F.shape[1]; h = n // 2
    S = np.fft.fft(F, axis=1); G = np.zeros((F.shape[0], n * U, F.shape[2]), np.complex128)
    G[:, :h] = S[:, :h]; G[:, -h + 1:] = S[:, h + 1:]; G[:, h] = 0.5 * S[:, h]; G[:, -h] = 0.5 * S[:, h]
    return (np.fft.ifft(G, axis=1) * U).astype(np.complex64)


# ----------------------------------------------------------------------------- geometry, DAS, masks
GX, GY, GZ = np.meshgrid(ROI_X, ROI_X, ROI_Z, indexing="ij")
VOX = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], 1); SHAPE = GX.shape        # 37 x 37 x 61 = 83 509 voxels


def das_matrix(m):
    """Sparse (Nvox, 1280*256) DAS operator: linear interpolation (2 taps) x demodulation phase, sign +1, uniform weights.
    Two-way time = (|voxel - virtual source| - d0) / c + |voxel - element| / c."""
    Nc, NsU = 256, 320 * U
    rows, cols, vals = [], [], []
    for i in range(0, len(VOX), 4000):
        v = VOX[i:i + 4000]
        tx = (np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2 + (v[:, 2] + m["apex"]) ** 2) - m["d0"]) / m["c"]
        tau = tx[:, None] + np.linalg.norm(v[:, None, :] - m["elem"][None], axis=2) / m["c"]
        fi = (tau - m["tinit"]) * m["fs"] * U
        ok = (fi >= 0) & (fi < NsU - 1)
        i0 = np.floor(fi).astype(np.int64); fr = fi - i0
        ph = np.exp(1j * 2 * np.pi * m["fd"] * tau)
        r = np.broadcast_to(np.arange(i, i + len(v))[:, None], fi.shape)
        ch = np.broadcast_to(np.arange(Nc)[None, :], fi.shape)
        for k, wgt in ((0, 1 - fr), (1, fr)):
            rows.append(r[ok]); cols.append(((i0 + k) * Nc + ch)[ok]); vals.append((wgt * ph)[ok])
    return sparse.csr_matrix((np.concatenate(vals).astype(np.complex64), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(VOX), NsU * Nc))


def das(A, Xf, threads):
    """Xf (1280*256, k) -> (Nvox, k), rows split over threads."""
    out = np.empty((A.shape[0], Xf.shape[1]), np.complex64)
    def job(r): out[r[0]:r[-1] + 1] = A[r[0]:r[-1] + 1] @ Xf
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(job, np.array_split(np.arange(A.shape[0]), threads * 4)))
    return out


def mvi_on_grid(m):
    """Vendor mvi volume ([z, y, x] on its own 0.6 mm grid) interpolated to the ROI voxels; NaN outside its support."""
    mz, my, mx = m["mvi_axes"]; mvi = m["mvi"]
    ci = np.stack([(VOX[:, 2] - mz[0]) / (mz[1] - mz[0]), (VOX[:, 1] - my[0]) / (my[1] - my[0]), (VOX[:, 0] - mx[0]) / (mx[1] - mx[0])])
    ok = np.isfinite(mvi)
    v = ndimage.map_coordinates(np.where(ok, mvi, 0.0), ci, order=1, cval=np.nan)
    okv = ndimage.map_coordinates(ok.astype(float), ci, order=1, cval=0) > 0.99
    return np.where(okv, v, np.nan)


def masks(ref):
    V = np.zeros(len(VOX), bool); G = np.zeros(len(VOX), bool); z = VOX[:, 2]
    for z0 in np.arange(ROI_Z[0], ROI_Z[-1] + 1e-9, SLAB):
        s = (z >= z0 - 1e-9) & (z < z0 + SLAB - 1e-9) & np.isfinite(ref)
        if s.sum() < 20:
            continue
        V |= s & (ref >= np.quantile(ref[s], 1 - TOP)); G |= s & (ref <= np.quantile(ref[s], BOTTOM))
    return V, G


def s_map(D):
    """D (nblk, Nvox, 2) band powers [B*, W*] -> smoothed S* = <D_B*>/<D_W*> - 1."""
    S = D[..., 0].mean(0) / D[..., 1].mean(0) - 1
    return ndimage.gaussian_filter(S.reshape(SHAPE), 1.0, mode="nearest").ravel()


def cnr(D, V, G):
    S = s_map(D)
    return float((S[V].mean() - S[G].mean()) / S[G].std())


def secondary(D, Dn, V):
    """noise floor change (ROI median, dB), vessel-mask noise and blood-band power change, blood excess change (dB)."""
    w, wn = D[..., 1].mean(0), Dn[..., 1].mean(0); b, bn = D[..., 0].mean(0), Dn[..., 0].mean(0)
    ex, exn = float(np.mean((b - w)[V])), float(np.mean((bn - wn)[V]))
    return dict(noise_floor_dB=float(np.median(10 * np.log10(w / wn))),
                noise_floor_vessel_dB=float(10 * np.log10(w[V].mean() / wn[V].mean())),
                blood_band_power_vessel_dB=float(10 * np.log10(b[V].mean() / bn[V].mean())),
                blood_excess_power_vessel_dB=float(10 * np.log10(ex / exn)) if ex > 0 and exn > 0 else float("nan"))


# ----------------------------------------------------------------------------- one recording
def cross_transform(data_dir, src):
    """full05 transform from all 8 blocks of another recording (cross-subject arm)."""
    m = load_meta(data_dir, src); _, bW, _ = bands(m["prf"])
    C = [noise_cov(slowfft(read_block(m, b), bW)) for b in range(NBLK)]
    return whitening(np.mean(C, 0))


def analyse(rec, data_dir, ndraw=0, T_cross=None, threads=8, batch=2):
    m = load_meta(data_dir, rec)
    bB, bW, fq = bands(m["prf"]); bins = np.concatenate([bB, bW]); nB, nb = len(bB), len(bB) + len(bW)
    log(f"{rec}: PRF {m['prf']:.4f} Hz; B* {nB} bins {np.abs(fq[bB]).min():.0f}-{np.abs(fq[bB]).max():.0f} Hz; "
        f"W* {len(bW)} bins {np.abs(fq[bW]).min():.0f}-{np.abs(fq[bW]).max():.0f} Hz")
    A = das_matrix(m)
    ref = mvi_on_grid(m); V, G = masks(ref)
    # pass 1: slow-time bins and per-block noise covariances; cross-fitted transforms
    F_all, C = [], []
    for b in range(NBLK):
        F = slowfft(read_block(m, b), bins); C.append(noise_cov(F[nB:])); F_all.append(F)
    S_par = {p: np.mean([C[j] for j in range(NBLK) if j % 2 != p], 0) for p in (0, 1)}   # used to whiten parity p
    T = {p: whitening(S_par[p]) for p in (0, 1)}
    crc = zlib.crc32(rec.encode()) % 100000
    T_rand = {(j, p): whitening(S_par[p], np.random.default_rng([SEED, crc, j, p])) for j in range(1, ndraw + 1) for p in (0, 1)}
    arms = ["nominal", "full05"] + (["cross"] if T_cross is not None else [])
    D = {a: np.zeros((NBLK, len(VOX), 2)) for a in arms}
    Dr = np.zeros((ndraw, 1, len(VOX), 2))      # random draws: block mean kept as a running sum (memory)
    log(f"{rec}: covariances done; {len(arms)} arms + {ndraw} random draws per block")

    def dens(X, Ts):
        """band powers for a list of transforms (None = identity), one DAS call."""
        Xf = np.empty((X.shape[1] * X.shape[2], nb * len(Ts)), np.complex64)     # column = (arm, bin), row = (sample, channel)
        for k, t in enumerate(Ts):
            Xf[:, k * nb:(k + 1) * nb] = (X if t is None else X @ t.T).reshape(nb, -1).T
        Y = das(A, Xf, threads); del Xf
        out = []
        for k in range(len(Ts)):
            P = np.abs(Y[:, k * nb:(k + 1) * nb].astype(np.complex128)) ** 2
            out.append(np.stack([P[:, :nB].mean(1), P[:, nB:].mean(1)], 1))
        return out

    for b in range(NBLK):
        X = upsample(F_all[b], m["tgc"]); F_all[b] = None; p = b % 2
        Ts = dict(nominal=None, full05=T[p], cross=T_cross)
        for a, d in zip(arms, dens(X, [Ts[a] for a in arms])):
            D[a][b] = d
        for j0 in range(0, ndraw, batch):
            js = list(range(j0, min(ndraw, j0 + batch)))
            for j, d in zip(js, dens(X, [T_rand[(j + 1, p)] for j in js])):
                Dr[j, 0] += d / NBLK
        log(f"{rec}: block {b} done")

    res = dict(recording=rec, prf=m["prf"], n_vessel_vox=int(V.sum()), n_background_vox=int(G.sum()), ndraw=ndraw,
               cnr_nominal=cnr(D["nominal"], V, G), cnr_full05=cnr(D["full05"], V, G))
    res["dCNR"] = res["cnr_full05"] - res["cnr_nominal"]
    res["secondary_full05"] = secondary(D["full05"], D["nominal"], V)
    if T_cross is not None:
        res["cnr_cross"] = cnr(D["cross"], V, G); res["dCNR_cross"] = res["cnr_cross"] - res["cnr_nominal"]
        res["secondary_cross"] = secondary(D["cross"], D["nominal"], V)
    res["rand_dCNR"] = [cnr(Dr[j], V, G) - res["cnr_nominal"] for j in range(ndraw)]
    res["rand_noise_floor_dB"] = [secondary(Dr[j], D["nominal"], V)["noise_floor_dB"] for j in range(ndraw)]
    if ndraw:
        res["frac_draws_below_full05"] = float(np.mean(np.array(res["rand_dCNR"]) < res["dCNR"]))
    maps = {a: s_map(D[a]).reshape(SHAPE).astype(np.float32) for a in arms}
    return res, maps


# ----------------------------------------------------------------------------- aggregation and pass rule
def boot_ci(v, seed):
    v = np.asarray(v, float); rng = np.random.default_rng(seed)
    bs = np.median(v[rng.integers(0, len(v), (PASS_RULE["boot_reps"], len(v)))], 1)
    return [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]


def aggregate(results):
    """Subject value = mean over its recordings. The verdict is computed only on the full held-out set (15 recordings)."""
    subj = {s: [results[r] for r in recs] for s, recs in HELD_OUT.items() if all(r in results for r in recs)}
    partial = sorted(s for s, recs in HELD_OUT.items() if any(r in results for r in recs) and s not in subj)
    out = dict(complete_subjects=list(subj), incomplete_subjects=partial)
    if not subj:
        out["verdict"] = "not computed (needs all 15 held-out recordings and the SP01-Left-2 control)"
        return out
    sv = lambda f: np.array([np.mean([f(r) for r in subj[s]]) for s in subj])
    d = sv(lambda r: r["dCNR"])
    out["per_subject_dCNR"] = dict(zip(subj, d.round(6).tolist()))
    out.update(median_dCNR=float(np.median(d)), ci90=boot_ci(d, PASS_RULE["boot_seed"]), n_positive=int((d > 0).sum()))
    for i, k in enumerate(["noise_floor_dB", "blood_excess_power_vessel_dB"]):
        v = np.array([np.nanmean([r["secondary_full05"][k] for r in subj[s]]) for s in subj]); v = v[np.isfinite(v)]
        out[k] = dict(median=float(np.median(v)), ci90=boot_ci(v, 20261025 + (0 if i == 0 else 3)), n_negative=int((v < 0).sum()), n=len(v))
    if all("dCNR_cross" in r for s in subj for r in subj[s]):
        dc = sv(lambda r: r["dCNR_cross"])
        out["cross"] = dict(per_subject_dCNR=dict(zip(subj, dc.round(6).tolist())), median_dCNR=float(np.median(dc)),
                            ci90=boot_ci(dc, PASS_RULE["boot_seed"]), n_positive=int((dc > 0).sum()))
    nd = min(r["ndraw"] for s in subj for r in subj[s])
    if nd:
        null = np.median(np.array([np.mean([np.array(r["rand_dCNR"][:nd]) for r in subj[s]], 0) for s in subj]), 0)
        out.update(n_draws=nd, p_random=float((1 + np.sum(null >= out["median_dCNR"])) / (1 + nd)),
                   null_median_dCNR_median=float(np.median(null)), null_median_dCNR_q95=float(np.quantile(null, 0.95)))
    ctrl = results.get(CONTROL)
    if len(subj) == len(HELD_OUT) and ctrl is not None:
        crit = dict(a=out["median_dCNR"] > 0 and out["ci90"][0] > 0, b=out["n_positive"] >= PASS_RULE["n_positive_min"],
                    c=nd > 0 and out["p_random"] <= PASS_RULE["p_random_max"], d=ctrl["dCNR"] <= PASS_RULE["control_dCNR_max"],
                    e=nd >= PASS_RULE["draws_required"])
        out["criteria"] = crit
        if not crit["e"]:
            out["verdict"] = f"INCONCLUSIVE (criterion c needs {PASS_RULE['draws_required']} draws; ran {nd})"
        elif all(crit.values()):
            out["verdict"] = "PASS"
        elif out["median_dCNR"] > 0 and crit["b"] and crit["c"] and crit["d"]:
            out["verdict"] = "FAIL(WIDE)"
        else:
            out["verdict"] = "FAIL"
    else:
        out["verdict"] = "not computed (needs all 15 held-out recordings and the SP01-Left-2 control)"
    return out


# ----------------------------------------------------------------------------- report
def print_table(results):
    print("\nrecording      CNR nom  CNR wht    dCNR  frozen   |diff|   cross  noise dB  blood dB  draws  frac<wht")
    for r in results.values():
        fz = FROZEN.get(r["recording"], {}); f = fz.get("dCNR")
        cross = f"{r['dCNR_cross']:+7.3f}" if "dCNR_cross" in r else "      -"
        frac = f"{r['frac_draws_below_full05']:.2f}" if r["ndraw"] else "-"
        print(f"{r['recording']:<13} {r['cnr_nominal']:8.3f} {r['cnr_full05']:8.3f} {r['dCNR']:+7.4f} "
              f"{(f'{f:+7.4f}' if f is not None else '      -')} {(f'{abs(r['dCNR'] - f):8.1e}' if f is not None else '       -')} "
              f"{cross} {r['secondary_full05']['noise_floor_dB']:+9.2f} {r['secondary_full05']['blood_excess_power_vessel_dB']:+9.2f} "
              f"{r['ndraw']:6d} {frac:>9}")
    print("dCNR = CNR(whitened) - CNR(nominal); frozen = the frozen confirmatory run; cross = whitening matrix from SP01-Left-2;\n"
          "noise dB = ROI-median noise-floor change; blood dB = vessel-mask blood excess (B* - W*) change; "
          "frac<wht = fraction of random draws with a lower dCNR.")


def make_figure(results, maps, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = list(results)
    fig = plt.figure(figsize=(12, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1])
    ax = fig.add_subplot(gs[0]); y = np.arange(len(recs))[::-1]
    ax.axvline(0, color="#898781", lw=0.8)
    ax.scatter([results[r]["dCNR"] for r in recs], y, s=50, color="#2a78d6", label="whitened (own session)", zorder=3)
    yc = [(results[r]["dCNR_cross"], yy) for r, yy in zip(recs, y) if "dCNR_cross" in results[r]]
    if yc:
        ax.scatter([c for c, _ in yc], [yy - 0.2 for _, yy in yc], s=40, marker="s", color="#eb6834",
                   label="whitened (matrix from another person, SP01-Left-2)", zorder=3)
    fz = [(FROZEN.get(r, {}).get("dCNR"), yy) for r, yy in zip(recs, y)]
    ax.scatter([f for f, _ in fz if f is not None], [yy for f, yy in fz if f is not None], s=110, facecolors="none",
               edgecolors="#0b0b0b", lw=1, label="frozen run", zorder=4)
    ax.set_yticks(y, recs); ax.set_xlabel("dCNR = CNR(whitened) - CNR(nominal)")
    ax.set_title("Vessel CNR gain per recording", loc="left", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1); ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    best = max(recs, key=lambda r: results[r]["dCNR"])
    mips = {a: maps[best][a].max(axis=1).T for a in ("nominal", "full05")}          # max over y -> (z, x)
    vmin, vmax = np.percentile(mips["nominal"], 1), np.percentile(mips["nominal"], 98)
    for k, (a, title) in enumerate((("nominal", "nominal"), ("full05", "whitened"))):
        axm = fig.add_subplot(gs[k + 1])
        im = axm.imshow(mips[a], cmap="gray", vmin=vmin, vmax=vmax, aspect="auto", origin="upper",
                        extent=[ROI_X[0] * 1e3, ROI_X[-1] * 1e3, ROI_Z[-1] * 1e3, ROI_Z[0] * 1e3])
        axm.set_title(f"{best}, S* max over y: {title}", loc="left", fontsize=10)
        axm.set_xlabel("x (mm)"); axm.set_ylabel("depth z (mm)" if k == 0 else "")
    fig.colorbar(im, ax=axm, shrink=0.8, label="S* (blood / noise - 1), same scale")
    fig.savefig(path, dpi=120); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default=os.path.join("results", "whitening"))
    ap.add_argument("--recordings", nargs="*", help="default: every held-out recording and the control found in --data-dir")
    ap.add_argument("--draws", type=int, default=0, help="random-covariance draws per recording (frozen run: 400). "
                    "Each draw costs about one extra arm, ~12 s per recording on a laptop CPU")
    ap.add_argument("--cross", default=CONTROL, help="recording whose all-block noise covariance gives the cross-subject "
                    "arm ('none' to skip; skipped with a note if it is not in --data-dir)")
    ap.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--batch", type=int, default=2, help="random draws per DAS call (each adds ~1 GB of peak memory)")
    a = ap.parse_args(argv)
    known = [r for recs in HELD_OUT.values() for r in recs] + [CONTROL] + DEV
    if a.recordings:
        bad = [r for r in a.recordings if r not in known]
        if bad:
            ap.error(f"{bad}: not analysable here ({EXCLUDED}); valid: {' '.join(known)}")
        recs = a.recordings
    else:
        recs = [r for r in known if os.path.exists(os.path.join(a.data_dir, r, "clip.h5")) and r not in DEV]
    if not recs:
        ap.error(f"no recordings in {a.data_dir}; run  python get_data.py --recordings SP06-Left SP01-Left-2 --refs")
    os.makedirs(a.out, exist_ok=True)
    T_cross = None
    if a.cross != "none":
        if os.path.exists(os.path.join(a.data_dir, a.cross, "clip.h5")):
            T_cross = cross_transform(a.data_dir, a.cross); log(f"cross-subject whitening matrix from {a.cross} (8 blocks)")
        else:
            log(f"note: {a.cross} not in {a.data_dir}; cross-subject arm skipped")
    results, maps = {}, {}
    for r in recs:
        t = time.time()
        results[r], maps[r] = analyse(r, a.data_dir, a.draws, None if r == a.cross else T_cross, a.threads, a.batch)
        results[r]["runtime_s"] = round(time.time() - t, 1)
        json.dump(results[r], open(os.path.join(a.out, f"{r}_whitening.json"), "w"), indent=1)
        log(f"{r}: dCNR {results[r]['dCNR']:+.4f} (frozen {FROZEN.get(r, {}).get('dCNR')}), {results[r]['runtime_s']:.0f} s")
    print_table(results)
    agg = aggregate(results)
    summary = {k: v for k, v in agg.items()}
    print("\nheld-out subjects complete:", ", ".join(agg["complete_subjects"]) or "none",
          "| incomplete:", ", ".join(agg["incomplete_subjects"]) or "none")
    if "median_dCNR" in agg:
        if len(agg["complete_subjects"]) < len(HELD_OUT):
            print(f"(partial set: {len(agg['complete_subjects'])} of {len(HELD_OUT)} held-out subjects complete; "
                  "the numbers below are not the study result)")
        print(f"median subject dCNR {agg['median_dCNR']:+.3f}, 90% CI [{agg['ci90'][0]:+.3f}, {agg['ci90'][1]:+.3f}], "
              f"{agg['n_positive']}/{len(agg['complete_subjects'])} positive"
              + (f"; p vs {agg['n_draws']} random draws = {agg['p_random']:.4f}" if "p_random" in agg else ""))
        print(f"noise floor {agg['noise_floor_dB']['median']:+.2f} dB; blood excess {agg['blood_excess_power_vessel_dB']['median']:+.2f} dB"
              + (f"; cross-subject matrix: median {agg['cross']['median_dCNR']:+.3f}" if "cross" in agg else ""))
    print("verdict:", agg["verdict"])
    json.dump(dict(recordings=results, aggregate=summary), open(os.path.join(a.out, "whitening_results.json"), "w"), indent=1)
    make_figure(results, maps, os.path.join(a.out, "whitening.png"))
    log(f"wrote {a.out}/whitening_results.json and whitening.png")


if __name__ == "__main__":
    main()
