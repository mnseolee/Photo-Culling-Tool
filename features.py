"""Person A: turn each photo into four scores -> features.csv

Usage (from the repo root):
    python features.py photos
"""
import sys, os
import cv2
import numpy as np
import pandas as pd


def load(path):
    # cv2.imread fails silently on Windows paths with non-ASCII characters,
    # so read the bytes ourselves and decode them.
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def score(path):
    img = load(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    img = cv2.resize(img, (1000, int(h * 1000 / w)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "filename": os.path.basename(path),
        "blur": cv2.Laplacian(gray, cv2.CV_64F).var(),
        "brightness": gray.mean(),
        "clipped": ((gray <= 5) | (gray >= 250)).mean(),
        "contrast": gray.std(),
    }


if len(sys.argv) != 2:
    sys.exit("Usage: python features.py <photo folder>")

folder = sys.argv[1]
files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
if not files:
    sys.exit(f"No .jpg/.jpeg/.png files found in {folder}")

rows, skipped = [], []
for f in files:
    row = score(os.path.join(folder, f))
    if row is None:
        skipped.append(f)
    else:
        rows.append(row)

pd.DataFrame(rows).to_csv("features.csv", index=False)
print(f"Wrote features.csv: {len(rows)} photos")
if skipped:
    print(f"Skipped {len(skipped)} unreadable file(s): {', '.join(skipped)}")
