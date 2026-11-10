"""Run the full photo-culling pipeline on a folder of photos -> picks/

Usage (from the repo root):
    python cull.py photos
"""
import os
import shutil
import subprocess
import sys
import joblib
import pandas as pd

if len(sys.argv) != 2:
    sys.exit("Usage: python cull.py <photo folder>")

folder = sys.argv[1]
if not os.path.isdir(folder):
    sys.exit(f"Error: folder '{folder}' does not exist.")

if not os.path.exists("model.joblib"):
    sys.exit("Error: 'model.joblib' not found. Run train.py first to save the trained model.")

# 1. Run feature extraction and burst grouping on the target folder
subprocess.run([sys.executable, "features.py", folder], check=True)
subprocess.run([sys.executable, "groups.py", folder, "groups_new.csv"], check=True)

# 2. Merge features and burst groups on filename
df = pd.read_csv("features.csv").merge(pd.read_csv("groups_new.csv"), on="filename")
if df.empty:
    sys.exit("Error: No matching photos found after merging features.csv and groups_new.csv.")

# 3. Compute exposure error and within-burst z-scores
df["exposure_err"] = (df["brightness"] - 128).abs()
feats = ["blur", "exposure_err", "clipped", "contrast"]

X = df.groupby("group_id")[feats].transform(
    lambda c: (c - c.mean()) / (c.std() + 1e-9)
).fillna(0)

# 4. Score every photo using the trained logistic regression model
model = joblib.load("model.joblib")
df["score"] = model.predict_proba(X)[:, 1]

# 5. Copy the top 3 photos per burst into picks/
os.makedirs("picks", exist_ok=True)
copied = 0
for _, grp in df.groupby("group_id"):
    for f in grp.nlargest(3, "score")["filename"]:
        shutil.copy(os.path.join(folder, f), os.path.join("picks", f))
        copied += 1

print(f"Done! Scored {len(df)} photos across {df['group_id'].nunique()} burst(s) -> copied {copied} photo(s) to picks/")
