#!/usr/bin/env python3
"""Convert an existing melody+accompaniment MIDI dataset to C major.

Usage (example):

    python convert_dataset_to_c_major.py \
        --mel_root /path/to/original/mel \
        --acc_root /path/to/original/acc \
        --out_root /path/to/c_major_dataset

After running this, point your training config at `--out_root` so that
ALL training data is already in C major on disk. This guarantees that
the model only ever sees C-major material during training.
"""

import os
import argparse
import pretty_midi

# === C-major normalization: ADDED ===
# Simple Krumhansl-Schmuckler style key estimation for major keys only.
# We detect the (major) key of the input, compute the semitone shift
# needed to move that key to C major, transpose the MIDI, and save.

C_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33,
                   4.38, 4.09, 2.52, 5.19,
                   2.39, 3.66, 2.29, 2.88]

KEY_NAMES_MAJOR = ["C", "C#", "D", "Eb", "E", "F",
                   "F#", "G", "Ab", "A", "Bb", "B"]


def estimate_major_key(pm: pretty_midi.PrettyMIDI):
    """Estimate major key (0=C,1=C#,...) and semitone shift to C.

    Returns:
        best_root (int): 0-11, where 0 means C major, 2 means D, etc.
        best_name (str): Human-readable major key name.
        shift_to_c (int): How many semitones to transpose UP to reach C major.
                          Negative means transpose down.
    """
    # Build pitch-class histogram
    pc_hist = [0.0] * 12
    for inst in pm.instruments:
        for note in inst.notes:
            pc = note.pitch % 12
            # Weight by duration so long notes count more
            duration = float(note.end - note.start)
            if duration <= 0.0:
                duration = 0.1
            pc_hist[pc] += duration

    # If no notes, assume C major and no shift
    if sum(pc_hist) == 0.0:
        return 0, "C", 0

    # Try all 12 major keys by rotating the C-major profile
    best_score = None
    best_root = 0
    for root in range(12):
        score = 0.0
        for i in range(12):
            pc = (root + i) % 12
            score += pc_hist[pc] * C_MAJOR_PROFILE[i]
        if best_score is None or score > best_score:
            best_score = score
            best_root = root

    # Semitone shift so that best_root -> C (0)
    # If best_root = G (7), we need -7 semitones to reach C.
    shift_to_c = (-best_root) % 12
    if shift_to_c > 6:
        # Prefer small negative shifts instead of large positives
        shift_to_c -= 12

    return best_root, KEY_NAMES_MAJOR[best_root], shift_to_c


def convert_pair_to_c_major(mel_path: str, acc_path: str, out_root: str):
    """Convert a melody+accompaniment MIDI pair into C major and save.

    - Detect key from melody file.
    - Transpose both mel and acc by the same amount.
    - Save to a mirrored directory tree under out_root.
    """
    rel_mel = os.path.relpath(mel_path)
    rel_acc = os.path.relpath(acc_path)

    out_mel_path = os.path.join(out_root, rel_mel)
    out_acc_path = os.path.join(out_root, rel_acc)

    os.makedirs(os.path.dirname(out_mel_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_acc_path), exist_ok=True)

    pm_mel = pretty_midi.PrettyMIDI(mel_path)
    root, name, shift_to_c = estimate_major_key(pm_mel)
    print(f"[C-major] {mel_path}: detected key {name}, shift {shift_to_c:+d} semitones to C")

    # transpose mel
    for inst in pm_mel.instruments:
        for note in inst.notes:
            note.pitch = max(0, min(127, note.pitch + shift_to_c))
    pm_mel.write(out_mel_path)

    # transpose acc with same shift
    pm_acc = pretty_midi.PrettyMIDI(acc_path)
    for inst in pm_acc.instruments:
        for note in inst.notes:
            note.pitch = max(0, min(127, note.pitch + shift_to_c))
    pm_acc.write(out_acc_path)


def batch_convert_dataset(mel_root: str, acc_root: str, out_root: str):
    """Walk through mel_root, and for each melody, find corresponding acc
    and convert both to C major.
    """
    for dirpath, _, filenames in os.walk(mel_root):
        for fn in filenames:
            if not fn.lower().endswith(".mid"):
                continue
            mel_path = os.path.join(dirpath, fn)
            rel = os.path.relpath(mel_path, mel_root)
            acc_path = os.path.join(acc_root, rel)
            if not os.path.isfile(acc_path):
                print(f"[WARN] acc file missing for {mel_path}, expected {acc_path}")
                continue
            convert_pair_to_c_major(mel_path, acc_path, out_root)


def main():
    parser = argparse.ArgumentParser(description="Convert a melody+accompaniment dataset to C major.")
    parser.add_argument("--mel_root", type=str, required=True, help="Original melody root directory")
    parser.add_argument("--acc_root", type=str, required=True, help="Original accompaniment root directory")
    parser.add_argument("--out_root", type=str, required=True, help="Output root directory (C-major)")
    args = parser.parse_args()

    batch_convert_dataset(args.mel_root, args.acc_root, args.out_root)


if __name__ == "__main__":
    main()
