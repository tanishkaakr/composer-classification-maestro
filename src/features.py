"""Step 1: turn every MAESTRO MIDI file into a vector of music-theory features.

Usage (from the project root):
    python src/features.py --root data/maestro-v3.0.0 --n-composers 6
    python src/features.py --root data/maestro-v3.0.0 --composers "Johann Sebastian Bach" "Frédéric Chopin"

Output: data/features.csv  (one row per MIDI file)

Feature groups (the column-name prefix is the group name, used for ablation):
    pitch_  pitch range/spread + pitch-class histogram
    mel_    melodic intervals (consecutive notes of the top voice)
    vert_   vertical intervals (notes sounding together) + dissonance ratio
    tex_    texture: polyphony, note density
    dyn_    dynamics (MIDI velocity)
    rhy_    rhythm: inter-onset intervals, note durations
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
from joblib import Parallel, delayed

FS = 10  # piano-roll frames per second (used for vertical intervals / polyphony)
DISSONANT = [1, 2, 6, 10, 11]  # interval classes: m2, M2, tritone, m7, M7
CONSONANT = [0, 3, 4, 5, 7, 8, 9]
IOI_EDGES = [0.03, 0.08, 0.12, 0.18, 0.25, 0.4, 0.6, 1.0, np.inf]


def _normalise(hist):
    total = hist.sum()
    return hist / total if total > 0 else hist.astype(float)


def _skyline(starts, pitches, tol=0.05):
    """Highest note of every chord/onset group = approximate melody line."""
    out, i, n = [], 0, len(starts)
    while i < n:
        j, best = i, pitches[i]
        while j + 1 < n and starts[j + 1] - starts[i] < tol:
            j += 1
            best = max(best, pitches[j])
        out.append(best)
        i = j + 1
    return np.array(out)


def extract_features(path):
    pm = pretty_midi.PrettyMIDI(str(path))
    notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    if len(notes) < 20:
        raise ValueError("too few notes")
    notes.sort(key=lambda n: (n.start, n.pitch))
    pitches = np.array([n.pitch for n in notes])
    starts = np.array([n.start for n in notes])
    ends = np.array([n.end for n in notes])
    vels = np.array([n.velocity for n in notes], dtype=float)
    durs = ends - starts
    total_time = max(ends.max() - starts.min(), 1e-6)
    f = {}

    # ---- pitch ----
    f["pitch_mean"] = pitches.mean()
    f["pitch_std"] = pitches.std()
    f["pitch_min"] = pitches.min()
    f["pitch_max"] = pitches.max()
    f["pitch_range"] = pitches.max() - pitches.min()
    pc = _normalise(np.bincount(pitches % 12, minlength=12))
    for i, v in enumerate(pc):
        f[f"pitch_pc_{i}"] = v

    # ---- melodic intervals (top voice) ----
    melody = _skyline(starts, pitches)
    d = np.abs(np.diff(melody))
    mh = _normalise(np.bincount(np.minimum(d, 12), minlength=13))  # bin 12 = 12+ semitones
    for i, v in enumerate(mh):
        f[f"mel_int_{i}"] = v
    f["mel_mean_abs"] = d.mean()
    f["mel_std"] = d.std()
    f["mel_leap_frac"] = (d >= 5).mean()
    f["mel_dissonant_frac"] = np.isin(d % 12, DISSONANT).mean()

    # ---- vertical intervals + texture (from a piano roll) ----
    roll = pm.get_piano_roll(fs=FS) > 0  # (128, frames) boolean
    counts = roll.sum(axis=0)
    vh = np.zeros(12)
    for t in np.nonzero(counts >= 2)[0]:
        p = np.nonzero(roll[:, t])[0]
        diffs = (p[None, :] - p[:, None])[np.triu_indices(len(p), 1)] % 12
        vh += np.bincount(diffs, minlength=12)
    for i, v in enumerate(_normalise(vh)):
        f[f"vert_int_{i}"] = v
    f["vert_dissonance_ratio"] = vh[DISSONANT].sum() / max(vh[CONSONANT].sum(), 1.0)

    active = counts[counts > 0]
    f["tex_poly_mean"] = active.mean() if len(active) else 0.0
    f["tex_poly_p95"] = np.percentile(active, 95) if len(active) else 0.0
    f["tex_poly_max"] = counts.max()
    f["tex_chord_frac"] = (counts >= 3).mean()
    f["tex_silence_frac"] = (counts == 0).mean()
    f["tex_note_density"] = len(notes) / total_time

    # ---- dynamics ----
    f["dyn_vel_mean"] = vels.mean()
    f["dyn_vel_std"] = vels.std()
    f["dyn_vel_change_std"] = np.diff(vels).std()

    # ---- rhythm ----
    ioi = np.diff(starts)
    ioi = ioi[ioi > 0.03]  # ignore near-simultaneous chord notes
    if len(ioi) < 5:
        ioi = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    f["rhy_ioi_mean"] = ioi.mean()
    f["rhy_ioi_median"] = np.median(ioi)
    f["rhy_ioi_std"] = ioi.std()
    f["rhy_ioi_cv"] = ioi.std() / ioi.mean()
    f["rhy_ioi_logstd"] = np.log(ioi).std()
    ih = _normalise(np.histogram(ioi, bins=IOI_EDGES)[0])
    for i, v in enumerate(ih):
        f[f"rhy_ioi_bin_{i}"] = v
    f["rhy_dur_mean"] = durs.mean()
    f["rhy_dur_std"] = durs.std()
    f["rhy_dur_cv"] = durs.std() / max(durs.mean(), 1e-6)
    return f


def _safe(root, rel_path):
    try:
        return extract_features(Path(root) / rel_path)
    except Exception as e:  # corrupt / empty file: skip it, but tell the user
        print(f"  skipped {rel_path}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/maestro-v3.0.0", help="folder with the CSV and year sub-folders")
    ap.add_argument("--csv", default=None, help="metadata CSV (default: <root>/maestro-v3.0.0.csv)")
    ap.add_argument("--composers", nargs="*", default=None, help="exact composer names to use")
    ap.add_argument("--n-composers", type=int, default=6, help="if --composers not given: take the N most frequent")
    ap.add_argument("--out", default="data/features.csv")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel workers (-1 = all cores)")
    args = ap.parse_args()

    csv_path = args.csv or str(Path(args.root) / "maestro-v3.0.0.csv")
    df = pd.read_csv(csv_path)
    df = df[~df["canonical_composer"].str.contains("/")]  # drop arrangements like "Bach / Busoni"
    counts = df["canonical_composer"].value_counts()
    print("Pieces per composer (top 15):\n", counts.head(15), "\n")

    composers = args.composers or list(counts.head(args.n_composers).index)
    missing = [c for c in composers if c not in counts.index]
    if missing:
        raise SystemExit(f"Composer(s) not found: {missing}. Copy names exactly from the list above.")
    df = df[df["canonical_composer"].isin(composers)].reset_index(drop=True)
    print(f"Using {len(composers)} composers, {len(df)} files:\n{df['canonical_composer'].value_counts()}\n")

    results = Parallel(n_jobs=args.jobs, verbose=5)(delayed(_safe)(args.root, p) for p in df["midi_filename"])
    rows = []
    for (_, meta), feats in zip(df.iterrows(), results):
        if feats is None:
            continue
        rows.append({"midi_filename": meta["midi_filename"], "composer": meta["canonical_composer"],
                     "title": meta["canonical_title"], "split": meta["split"], **feats})
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nSaved {len(out)} rows x {out.shape[1] - 4} features to {args.out}")
    print(pd.crosstab(out["composer"], out["split"]))


if __name__ == "__main__":
    main()
