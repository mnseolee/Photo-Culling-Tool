"""Step 7: open the 5 blurriest photos so you can check them by eye.

Usage:
    python sanity_check.py photos
"""
import sys, os
import pandas as pd

folder = sys.argv[1] if len(sys.argv) > 1 else "photos"
df = pd.read_csv("features.csv")

print(df.describe().round(3), "\n")
worst = df.sort_values("blur").head(5)
print("5 lowest blur scores (should be the blurry ones):")
print(worst[["filename", "blur"]].to_string(index=False))

for f in worst["filename"]:
    os.startfile(os.path.join(folder, f))  # Windows: opens in the default viewer
