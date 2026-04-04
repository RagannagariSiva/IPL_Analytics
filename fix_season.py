# Run this once to patch both files in-place
import re

# ── 1. features.py: encode season before storing in ml_dataset ──────────────
with open("ml/features.py", "r") as f:
    src = f.read()

old = '"season":                 m["season"],'
new = '"season":                 int(str(m["season"]).split("/")[0]),'

if old in src:
    src = src.replace(old, new)
    with open("ml/features.py", "w") as f:
        f.write(src)
    print("✅ features.py patched")
else:
    print("⚠️  features.py – target line not found (may already be patched)")

# ── 2. predictor.py: ensure X is float before feeding to sklearn ─────────────
with open("ml/predictor.py", "r") as f:
    src = f.read()

old2 = '    X = ml_df[FEATURE_COLS].fillna(0.0)'
new2 = ('    X = ml_df[FEATURE_COLS].copy()\n'
        '    # normalise season: "2009/10" -> 2009\n'
        '    if "season" in X.columns:\n'
        '        X["season"] = X["season"].apply(\n'
        '            lambda s: int(str(s).split("/")[0]) if pd.notna(s) else 2008)\n'
        '    X = X.fillna(0.0).astype(float)')

if old2 in src:
    src = src.replace(old2, new2)
    with open("ml/predictor.py", "w") as f:
        f.write(src)
    print("✅ predictor.py patched")
else:
    print("⚠️  predictor.py – target line not found (may already be patched)")

print("\nDone. Now run: python train.py")
