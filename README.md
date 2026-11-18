# ECG Heartbeat Classification using 1D-CNN

Classify heartbeats from ECG signals into 5 AAMI classes using MIT-BIH Arrhythmia Database.

## Results
- **Test Accuracy: 96.94%** (15 Epochs, CPU)
- Train: 83,648 beats | Test: 20,913 beats | Total: 104,061 beats built from PhysioNet

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| N - Normal | 1.00 | 0.97 | 0.99 | 17556 |
| S - Supraventricular | 0.63 | 0.93 | 0.75 | 537 |
| V - Ventricular | 0.95 | 0.97 | 0.96 | 1444 |
| F - Fusion | 0.40 | 0.92 | 0.56 | 160 |
| Q - Unknown | 0.99 | 1.00 | 0.99 | 1196 |

## Why F is low?
Only 802 F beats in full data (class imbalance). Model rarely predicts F, but when it does it is often wrong - classic imbalance problem. Handled with balanced class weights.

## Architecture
1D-CNN: Conv1d(1->32) + BatchNorm + MaxPool -> Conv1d(32->64) -> Conv1d(64->128) -> FC(128*23 -> 128) -> FC(128 -> 5)
Input: 187 samples per beat (90 before R-peak + 97 after)

## How to Run
```bash
pip install -r requirements.txt
python src/preprocess.py  # Builds dataset from PhysioNet via wfdb
python src/train.py
python src/evaluate.py
```

## Dataset
Built from PhysioNet MIT-BIH Arrhythmia Database (records 100-234) using wfdb. No Kaggle/GitHub CSV needed - original source used directly.

## Files
- `src/preprocess.py` - PhysioNet download + beat segmentation
- `src/model.py` - 1D-CNN
- `src/train.py` - Training with balanced class weights
- `src/evaluate.py` - Report + Confusion Matrix
- `results/confusion_matrix.png` - Confusion Matrix
