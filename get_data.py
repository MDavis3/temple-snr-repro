# SPDX-License-Identifier: MIT
"""Download only the pre-bolus (contrast-free) clip of Open-H-RF / Resolve Stroke clinical recordings.

Each published HDF5 file is ~5.3 GB (five 1-s clips). We read the file's HDF5 index over HTTP range
requests, then fetch only the compressed chunks of
  * clip 'baseline_minus1s' (4000 frames, before microbubble arrival)  -> what we measure
  * the first 256 frames of 'baseline_plus10s' (contrast present)      -> contamination check only
(about 0.9-1.0 GB per recording). Chunks are copied byte-for-byte into a small local HDF5 (no re-compression).

Verification per recording: (a) every chunk decodes to a full (320, 256, 2) int16 frame; (b) every chunk stored
in clip.h5 is byte-identical to the downloaded bytes; (c) 32 random chunks fetched again from the server are
byte-identical; (d) the server's file size equals the size the index was read from. sha256 of every chunk is
written to manifest.json (compare yours with someone else's).
Resumable: downloaded pieces are kept until the recording verifies; rerun the same command to continue.

usage: python get_data.py [--n 4 | --all | --recordings SP01-Left-1 ...] [--data-dir data]
  --n K   the first K names of RECORDINGS (K = 1..20): K <= 10 gives one recording per subject (the first listed),
          K = 20 is the full study (10 subjects x 2 recordings, about 19 GB). --all is the same as --n 20.
"""
import argparse, concurrent.futures as cf, hashlib, io, json, os, random, shutil, sys, time
import urllib.error, urllib.request

import numpy as np
import h5py
import hdf5plugin  # noqa: F401  (registers the Blosc filter used by the dataset)

# ----------------------------------------------------------------------------- configuration
URL = "https://huggingface.co/datasets/nvidia/Open-H-RF/resolve/main/resolvestroke/clinical/{0}/{0}.hdf5"
# All 20 clinical acquisitions (10 subjects x 2). FIRST = the first listed per subject (fixed before any data was
# looked at); SECOND = each subject's other acquisition. --n K takes the first K of FIRST + SECOND.
FIRST = ["SP01-Left-1", "SP02-Left-1", "SP03-Left", "SP04-Left-1", "SP05-Left",
         "SP06-Left", "SP07-Left", "SP08-Left", "SP09-Left-1", "SP10-Left"]
SECOND = ["SP01-Left-2", "SP02-Left-2", "SP03-Right", "SP04-Left-2", "SP05-Right",
          "SP06-Right", "SP07-Right", "SP08-Right", "SP09-Left-2", "SP10-Right"]
RECORDINGS = FIRST + SECOND
CLIP_LABEL, CONTROL_LABEL, N_CONTROL = "baseline_minus1s", "baseline_plus10s", 256
RAW = "tracks/track_0/data/raw_data"
PIECE_BYTES = 32 << 20      # download unit
WORKERS = 4                 # parallel range requests
N_REFETCH = 32              # random chunks re-fetched for byte comparison
# ---------------------------------------------------------------------------------------------


def fetch(url, lo, hi, tries=6):
    """Bytes lo..hi (inclusive) of url, with retries; length is checked."""
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}", "User-Agent": "temple-snr-repro"})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = r.read()
            if len(d) == hi - lo + 1:
                return d
            raise IOError(f"short read {len(d)} != {hi - lo + 1}")
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500 and e.code != 429:   # not found / forbidden: retrying will not help
                raise SystemExit(f"HTTP {e.code} for {url}")
            if t == tries - 1:
                raise
            time.sleep(3 * (t + 1))
        except Exception:  # network hiccup or expired redirect: retry from the canonical URL
            if t == tries - 1:
                raise
            time.sleep(3 * (t + 1))


class RemoteFile(io.RawIOBase):
    """Read-only, seekable view of a remote file (1 MiB block cache), for h5py's file-object driver."""
    def __init__(self, url, block=1 << 20):
        self.url, self.block, self.pos, self.cache = url, block, 0, {}
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "temple-snr-repro"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].split("/")[-1])
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos
    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        out, p, end = bytearray(), self.pos, self.pos + n
        while p < end:
            b = p // self.block
            if b not in self.cache:
                lo = b * self.block
                self.cache[b] = fetch(self.url, lo, min(self.size, lo + self.block) - 1)
            d = self.cache[b]; o = p - b * self.block; take = min(len(d) - o, end - p)
            out += d[o:o + take]; p += take
        buf[:n] = out; self.pos = end
        return n


def read_index(url, out_dir):
    """Chunk byte offsets of the wanted frames + small metadata datasets (no raw samples)."""
    rf = RemoteFile(url)
    with h5py.File(rf, "r") as f:
        ds = f[RAW]
        labels = [x.decode() if isinstance(x, bytes) else str(x) for x in f["metadata/annotations/label"][()]]
        frames = {"clip": [i for i, l in enumerate(labels) if l == CLIP_LABEL],
                  "control": [i for i, l in enumerate(labels) if l == CONTROL_LABEL][:N_CONTROL]}
        assert len(frames["clip"]) == 4000 and len(frames["control"]) == N_CONTROL, {k: len(v) for k, v in frames.items()}
        assert ds.chunks[0] == 1 and ds.chunks[2:] == ds.shape[2:], "expected one chunk per frame"
        chunks = {}
        for part, fr in frames.items():
            chunks[part] = []
            for k in fr:
                ci = ds.id.get_chunk_info_by_coord((k,) + (0,) * (ds.ndim - 1))
                chunks[part].append([k, int(ci.byte_offset), int(ci.size), int(ci.filter_mask)])
        meta = {}
        for name in ["probe/probe_geometry", "metadata/subject/id"] + [
                "tracks/track_0/scan/" + k for k in ["tgc_gain_curve", "initial_times", "sampling_frequency",
                "sound_speed", "focus_distances", "center_frequency", "demodulation_frequency", "time_to_next_transmit"]]:
            a = f[name][()]
            meta[name] = a.decode() if isinstance(a, bytes) else np.asarray(a).tolist()
        index = dict(url=url, file_size=rf.size, raw_shape=list(ds.shape), raw_dtype=str(ds.dtype),
                     filters={str(k): list(v) for k, v in ds._filters.items()}, chunks=chunks, meta=meta,
                     description=str(f.attrs.get("description", "")))
    json.dump(index, open(os.path.join(out_dir, "index.json"), "w"))
    return index


def plan_pieces(chunk_list):
    """Group consecutive chunks into download pieces of about PIECE_BYTES; each chunk lies in one piece."""
    pieces, cur = [], []
    for c in sorted(chunk_list, key=lambda c: c[1]):
        if cur and (c[1] + c[2] - cur[0][1]) > PIECE_BYTES:
            pieces.append(cur); cur = []
        cur.append(c)
    if cur:
        pieces.append(cur)
    return [(p[0][1], max(c[1] + c[2] for c in p) - 1, p) for p in pieces]


def download_pieces(url, pieces, part_dir, tag):
    os.makedirs(part_dir, exist_ok=True)
    def one(j):
        lo, hi, _ = pieces[j]; fn = os.path.join(part_dir, f"{tag}_{j:04d}.bin")
        if os.path.exists(fn) and os.path.getsize(fn) == hi - lo + 1:
            return 0                                # already have it (resume)
        d = fetch(url, lo, hi)
        with open(fn + ".tmp", "wb") as fh:
            fh.write(d)
        os.replace(fn + ".tmp", fn)                 # atomic: a piece is either complete or absent
        return len(d)
    got, t0 = 0, time.time()
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for i, n in enumerate(ex.map(one, range(len(pieces)))):
            got += n
            print(f"\r  {tag}: piece {i + 1}/{len(pieces)}  {got / 1e6:8.1f} MB new  {got / 1e6 / max(time.time() - t0, 1e-3):6.1f} MB/s",
                  end="", flush=True)
    print()


def build_and_verify(name, index, rec_dir, n_refetch):
    part_dir = os.path.join(rec_dir, "parts")
    out = os.path.join(rec_dir, "clip.h5"); tmp = out + ".tmp"
    shape = index["raw_shape"]; frame_shape = tuple(shape[2:]) if shape[1] == 1 else tuple(shape[1:])
    filt = index["filters"]["32001"]                                     # Blosc parameters of the source
    manifest = {}
    with h5py.File(tmp, "w") as f:
        for part, chunk_list in index["chunks"].items():
            ds = f.create_dataset(part, shape=(len(chunk_list),) + frame_shape, dtype=index["raw_dtype"],
                                  chunks=(1,) + frame_shape, compression=32001, compression_opts=tuple(filt))
            ds.attrs["source_frames"] = [c[0] for c in chunk_list]
            pos = {c[0]: j for j, c in enumerate(chunk_list)}
            for p, (lo, hi, cl) in enumerate(plan_pieces(chunk_list)):
                buf = open(os.path.join(part_dir, f"{part}_{p:04d}.bin"), "rb").read()
                for k, off, size, mask in cl:
                    b = buf[off - lo: off - lo + size]
                    ds.id.write_direct_chunk((pos[k],) + (0,) * len(frame_shape), b, filter_mask=mask)
                    manifest[str(k)] = hashlib.sha256(b).hexdigest()
        g = f.create_group("meta")
        for k, v in index["meta"].items():
            g.create_dataset(k.replace("/", "__"), data=v)
        f.attrs["source_url"] = index["url"]; f.attrs["recording"] = name
        f.attrs["licence"] = "CC BY 4.0, Open-H-RF / Resolve Stroke (modified: subset of frames)"
    # (a) every chunk decodes to a full frame; (b) stored chunk bytes == downloaded bytes
    bad, byte_bad = 0, 0
    with h5py.File(tmp, "r") as f:
        for part, cl in index["chunks"].items():
            ds = f[part]
            for j, c in enumerate(cl):
                fm, b = ds.id.read_direct_chunk((j,) + (0,) * len(frame_shape))
                byte_bad += hashlib.sha256(b).hexdigest() != manifest[str(c[0])] or fm != c[3]
            for j in range(0, ds.shape[0], 200):
                try:
                    x = ds[j:j + 200]; assert x.shape[1:] == frame_shape
                except Exception as e:
                    bad += 1; print("  DECODE FAIL", part, j, e)
    # (c) random chunks re-fetched from the server must be byte-identical
    allc = [c for cl in index["chunks"].values() for c in cl]
    pick = random.Random(name).sample(allc, min(n_refetch, len(allc)))
    mism = [c[0] for c in pick if hashlib.sha256(fetch(index["url"], c[1], c[1] + c[2] - 1)).hexdigest() != manifest[str(c[0])]]
    size_ok = RemoteFile(index["url"]).size == index["file_size"]                      # (d)
    ok = bad == 0 and byte_bad == 0 and not mism and size_ok
    rep = dict(recording=name, ok=ok, chunks=len(manifest), decode_fail_blocks=bad, stored_byte_mismatch=int(byte_bad),
               remote_size_ok=size_ok, refetched=len(pick),
               refetch_mismatch=mism, bytes=sum(c[2] for c in allc))
    json.dump(dict(rep, chunk_sha256=manifest), open(os.path.join(rec_dir, "manifest.json"), "w"), indent=0)
    if ok:
        os.replace(tmp, out)
    return rep


def get_recording(name, data_dir, n_refetch=N_REFETCH, keep_pieces=False):
    rec_dir = os.path.join(data_dir, name); os.makedirs(rec_dir, exist_ok=True)
    if os.path.exists(os.path.join(rec_dir, "clip.h5")):
        print(f"{name}: already downloaded and verified"); return json.load(open(os.path.join(rec_dir, "manifest.json")))
    ip = os.path.join(rec_dir, "index.json")
    index = json.load(open(ip)) if os.path.exists(ip) else read_index(URL.format(name), rec_dir)
    for part, cl in index["chunks"].items():
        download_pieces(index["url"], plan_pieces(cl), os.path.join(rec_dir, "parts"), part)
    rep = build_and_verify(name, index, rec_dir, n_refetch)
    print(f"{name}: {'VERIFIED' if rep['ok'] else 'FAILED'}  chunks={rep['chunks']} decode_fail={rep['decode_fail_blocks']} "
          f"stored_mismatch={rep['stored_byte_mismatch']} size_ok={rep['remote_size_ok']} "
          f"refetch {rep['refetched'] - len(rep['refetch_mismatch'])}/{rep['refetched']} identical, {rep['bytes'] / 1e9:.2f} GB")
    if rep["ok"] and not keep_pieces:
        shutil.rmtree(os.path.join(rec_dir, "parts"))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=4, help="first N names of the fixed list (default 4; 10 = one per subject; 20 = all)")
    ap.add_argument("--all", action="store_true", help="all 20 recordings (same as --n 20, about 19 GB)")
    ap.add_argument("--recordings", nargs="*", help="explicit recording names (overrides --n and --all)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--refetch", type=int, default=N_REFETCH)
    ap.add_argument("--keep-pieces", action="store_true")
    a = ap.parse_args(argv)
    if a.recordings:
        bad = [n for n in a.recordings if n not in RECORDINGS]
        if bad:
            ap.error(f"unknown recording(s) {bad}; valid names: {' '.join(RECORDINGS)}")
        names = a.recordings
    else:
        n = len(RECORDINGS) if a.all else a.n
        if not 1 <= n <= len(RECORDINGS):
            ap.error(f"--n must be between 1 and {len(RECORDINGS)} (got {n})")
        names = RECORDINGS[:n]
    print(f"{len(names)} recording(s): {' '.join(names)}")
    t0 = time.time()
    try:
        reps = [get_recording(n, a.data_dir, a.refetch, a.keep_pieces) for n in names]
    except KeyboardInterrupt:
        print("\ninterrupted; finished pieces are kept - rerun the same command to resume"); return 130
    print(f"done in {(time.time() - t0) / 60:.1f} min; verified {sum(r['ok'] for r in reps)}/{len(reps)}")
    return 0 if all(r["ok"] for r in reps) else 1


if __name__ == "__main__":
    sys.exit(main())
