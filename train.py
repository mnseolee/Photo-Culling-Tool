import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("features.csv").merge(pd.read_csv("labels.csv"), on="filename")
df["exposure_err"] = (df.brightness - 128).abs()
feats = ["blur", "exposure_err", "clipped", "contrast"]

# z-score within each burst (fillna handles one-photo groups)
df[feats] = df.groupby("group_id")[feats].transform(
    lambda c: (c - c.mean()) / (c.std() + 1e-9)).fillna(0)

hits = {"random": [], "formula": [], "ml": []}
for g in df.group_id.unique():
    train, test = df[df.group_id != g], df[df.group_id == g]
    model = LogisticRegression().fit(train[feats], train.label)
    ml_pick = test.iloc[model.predict_proba(test[feats])[:, 1].argmax()]
    formula = test.blur - test.clipped - test.exposure_err
    f_pick = test.loc[formula.idxmax()]
    hits["ml"].append(ml_pick.label == 1)
    hits["formula"].append(f_pick.label == 1)
    hits["random"].append(test.label.mean())   # chance a random pick is a keeper

for name, v in hits.items():
    print(f"{name}: {100 * np.mean(v):.0f}% of groups, top pick was a keeper")

final = LogisticRegression().fit(df[feats], df.label)
print(dict(zip(feats, final.coef_[0].round(2))))   # which feature mattered most