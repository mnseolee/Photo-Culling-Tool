# Photo-Culling-Tool

# Photo-Culling-Tool

Picks the best 3 photos from each burst of near-identical shots, so you review 3 instead of 10.

Built in one day by two first-year Nanotechnology Engineering students at the University of Waterloo, as a learning project in computer vision and basic ML.

## How it works

1. **Group bursts by capture time** (`groups.py`). Photos taken less than 12 seconds apart belong to the same burst. The time comes from each JPEG's EXIF data.
2. **Score every photo** (`features.py`) on four numbers:
   - **blur**: edge crispness, the strength of the strongest 1% of edges (Sobel filter) divided by the brightness range
   - **brightness**: average grey level, 0–255
   - **clipped**: fraction of pixels that are pure black or pure white
   - **contrast**: standard deviation of grey levels
3. **Compare each photo to its own burst.** Every score is converted to a z-score within its burst, so a night scene isn't compared to a day scene.
4. **Rank and copy** (`cull.py`). Score = crispness − clipping − exposure error. The top 3 per burst are copied to `picks/`.

## Results

Dataset: 122 phone photos in 13 bursts, labelled by hand (19 keepers). The metric is the % of bursts where the method's top pick was one of our keepers. Each burst was tested without being used for training (leave-one-group-out).

| Run | Sharpness measure | Random | Formula | Logistic regression |
| --- | --- | --- | --- | --- |
| 1 | Laplacian variance, 1000 px | 16% | 8% | 8% |
| 2 | Laplacian variance, 250 px | 16% | 15% | 15% |
| 3 | Edge crispness | 16% | **38%** | 15% |

Run 3's formula picked a keeper in 5 of 13 bursts. A random picker does that about 4% of the time, or roughly 11% once you account for our three attempts. **It's a promising signal, not proof.** Before running it, we agreed run 3 would be the last attempt and that it needed at least 5 of 13 to count as working.

## What we learned

- **Perceptual hashing couldn't find the bursts.** Consecutive photos in the same burst differed by 20–36 bits, as much as unrelated scenes, because we deliberately varied each shot. Grouping by capture time worked immediately.
- **The standard blur score measures scene detail, not focus.** Tilting the camera toward textured ground raised the Laplacian score even though nothing got sharper. Example: in one burst, it rated a shot pointing more at the ground as "sharpest" over our clean keeper.
- **Combining features mattered more than any single one.** Edge crispness alone matched our keeper in only 2 of 13 bursts; together with the exposure and clipping checks it matched 5.
- **The hand-written formula beat machine learning.** With only 19 keepers, logistic regression couldn't learn stable weights (it flipped the sign of exposure error between runs). A formula with sensible signs needs no data.

## Limitations

- Small dataset: 122 photos, 13 bursts, one labeller. One burst moves the result by about 8 points.
- **Easier than real culling.** In each burst we took a clean shot first, then deliberately ruined the rest (shaking, darkening, reframing). In 11 of 13 bursts the keeper was the first photo. We did **not** use photo position as a feature, since it would only work because of how we shot the data.
- We tried three sharpness measures on the same 13 bursts, so the run-3 result could partly be luck. The real test is a new set of bursts.
- JPEG only. No RAW files, faces or composition.

## How to run

```
pip install opencv-python numpy pandas pillow imagehash scikit-learn
python cull.py photos
```

The best 3 photos per burst appear in `picks/`.

To reproduce the results table: run `python features.py photos`, then `python groups.py photos` and label `labels.csv` (1 = keeper), then `python train.py`. Running `groups.py` again overwrites your labels.

## Files

| File | What it does |
| --- | --- |
| `features.py` | Scores every photo, writes `features.csv` |
| `groups.py` | Groups bursts by capture time, writes `labels.csv` |
| `train.py` | Compares random, formula and logistic regression |
| `compare.py` | Shows each method's pick next to ours, per burst |
| `cull.py` | The tool: copies the top 3 per burst to `picks/` |
