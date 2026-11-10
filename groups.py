import sys, os
import imagehash
import pandas as pd
from PIL import Image

folder = sys.argv[1]
THRESHOLD = 12   # bigger = fewer, larger groups
files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))

rows, group, prev = [], 0, None
for f in files:
    h = imagehash.phash(Image.open(os.path.join(folder, f)))
    if prev is not None and h - prev > THRESHOLD:
        group += 1
    rows.append({"filename": f, "group_id": group, "label": 0})
    prev = h
out = sys.argv[2] if len(sys.argv) > 2 else "labels.csv"   # never overwrite your labels
pd.DataFrame(rows).to_csv(out, index=False)