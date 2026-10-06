# Composer Classification from Symbolic Music (MAESTRO v3.0.0)

Classifies piano pieces by composer from MIDI files using music-theory features
(pitch, melodic intervals, vertical intervals/dissonance, texture, dynamics, rhythm)
and compares Logistic Regression, SVM, Gradient Boosting and an MLP.
Inspired by "Analysis and Classification of Symbolic Western Classical Music by Composer".

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data (not committed)
Download from https://magenta.tensorflow.org/datasets/maestro :
- `maestro-v3.0.0-midi.zip`
- `maestro-v3.0.0.csv`

Unzip so the layout is:
```
data/maestro-v3.0.0/maestro-v3.0.0.csv
data/maestro-v3.0.0/2004/*.midi ... data/maestro-v3.0.0/2018/*.midi
```

## Run
```bash
python src/features.py --root data/maestro-v3.0.0 --n-composers 6   # -> data/features.csv
python src/train.py                                                  # -> results/, models/
python src/predict.py data/maestro-v3.0.0/2004/<some_file>.midi      # demo
```
Choose composers explicitly with `--composers "Johann Sebastian Bach" "Frédéric Chopin" ...`.

## Method summary
- Split: the MAESTRO-provided train/validation/test split (no composition appears in two splits).
- Hyper-parameters tuned on validation macro-F1 only; test set evaluated once per model.
- Metrics: accuracy, macro precision/recall/F1, confusion matrix.
- Analysis: feature-group ablation and permutation feature importance.

## Authors
<team member names / SRNs>
