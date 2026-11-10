import os, shutil, subprocess, sys
import pandas as pd

if len(sys.argv) != 2:
    sys.exit("Usage: python cull.py <photo folder>")
folder = sys.argv[1]

# 1. score every photo and group the bursts (same scripts as training)
subprocess.run([sys.executable, "features.py", folder], check=True)
subprocess.run([sys.executable, "groups.py", folder, "groups_new.csv"], check=True)

df = pd.read_csv("features.csv").merge(pd.read_csv("groups_new.csv"), on="filename")
df["exposure_err"] = (df.brightness - 128).abs()
feats = ["blur", "exposure_err", "clipped", "contrast"]

# 2. compare each photo to the rest of its burst (z-scores within group)
X = df.groupby("group_id")[feats].transform(
    lambda c: (c - c.mean()) / (c.std() + 1e-9)).fillna(0)

# 3. the winning formula from run 3: crisp, not clipped, well exposed
df["score"] = X.blur - X.clipped - X.exposure_err

# 4. copy the top 3 of each burst into a fresh picks/ folder
shutil.rmtree("picks", ignore_errors=True)
os.makedirs("picks")
for g, grp in df.groupby("group_id"):
    top = grp.nlargest(3, "score")
    for f in top.filename:
        shutil.copy(os.path.join(folder, f), "picks")
    print(f"group {g:2d}: kept {', '.join(top.filename)}")

print(f"Copied {len(os.listdir('picks'))} photos into picks/")