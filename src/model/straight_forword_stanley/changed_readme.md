# File-by-File Change Summary

This section describes which files were modified, what was added, and why each change was necessary to support C-major-only training and key-agnostic inference.

---

## 1. New File: `convert_dataset_to_c_major.py`

### Purpose
Provide an offline preprocessing tool that converts the entire training dataset into C major.

### What was added
- Key detection using a major-key pitch profile.
- Computation of semitone shift needed to move the melody to C major.
- Transposition of both melody and accompaniment MIDI files.
- Output directory mirroring the original dataset structure.

### Why
The model must be trained only on C-major-normalized data to ensure consistent token distributions and remove key-based variability.

---

## 2. Modified File: `inference.py`

### What was added
- Key detection utilities using pitch-class statistics.
- `make_c_major_pair` function to:
  - detect the input melody’s key,
  - compute shift to C major,
  - create temporary C-major melody and accompaniment files.
- Updated preprocessing to operate only on the C-major files.
- Added `transpose_semitones` parameter to `decode_output`:
  - allows the generated C-major output to be shifted back into the input melody’s original key.

### Why
Inference must run entirely in C major for consistency with training.  
However, users expect the final accompaniment in the original key, so the code must shift the output back after generation.

---

## 3. Modified File: `inference_fake_realtime.py`

### What was added
- Same key detection and transposition logic as in `inference.py`.
- C-major conversion of each input segment before processing.
- Updated decoding to transpose final output back into original key.
- Use of zero pitch shift during preprocessing (key handling is done directly on MIDI files instead).

### Why
Real-time inference must behave exactly like offline inference, ensuring:
- consistent C-major internal processing,
- correct key restoration for real-time generation.

---

## 4. Unmodified File: `model.py`

### What changed
- **No changes were made** to this file.

### Why
The model architecture, training step, and loss computation do not depend on key information.  
All key normalization is performed:
- before preprocessing (MIDI-level shift),
- and after decoding (reverse shift).

Therefore, model-level code remains unchanged and continues to operate on tokenized C-major data only.

---

## 5. Unmodified File: `config.py`

### What changed
- No modifications were required.

### Why
Key normalization is performed about the dataset and inference input paths, not inside configuration logic.

