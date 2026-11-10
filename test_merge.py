"""Setup step 4: 5-row fake merge test. Writes fake_features.csv and
fake_labels.csv (so the real features.csv is never overwritten) and merges them.

Usage:
    python test_merge.py
"""
import pandas as pd

names = [f"IMG_{i:04d}.jpg" for i in range(1, 6)]

features = pd.DataFrame({
    "filename": names,
    "blur": [120.5, 15.2, 340.0, 88.1, 210.7],
    "brightness": [110.0, 45.3, 200.1, 128.0, 95.6],
    "clipped": [0.01, 0.20, 0.35, 0.00, 0.03],
    "contrast": [52.1, 20.4, 61.0, 48.8, 55.2],
})
labels = pd.DataFrame({
    "filename": names,
    "group_id": [1, 1, 1, 2, 2],
    "label": [1, 0, 0, 1, 0],
})
features.to_csv("fake_features.csv", index=False)
labels.to_csv("fake_labels.csv", index=False)

merged = pd.merge(pd.read_csv("fake_features.csv"), pd.read_csv("fake_labels.csv"), on="filename")
print(merged)
print(f"\n{len(merged)} rows ->", "PASS" if len(merged) == 5 else "FAIL")
