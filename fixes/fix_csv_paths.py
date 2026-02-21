"""
Fix stale base paths in split CSVs.
Old root: D:\M3 Projects\DeepFake_Research
New root: D:\M3\DeepFake_Research
"""
import csv
import os

OLD = r'D:\M3 Projects\DeepFake_Research'
NEW = r'D:\M3\DeepFake_Research'
SPLITS = r'D:\M3\DeepFake_Research\outputs\splits'

csv_files = sorted(f for f in os.listdir(SPLITS) if f.endswith('.csv'))

# ── Step 1: Fix any remaining broken paths ────────────────────────────
print("Checking / fixing stale paths ...")
for fname in csv_files:
    fpath = os.path.join(SPLITS, fname)
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    count = content.count(OLD)
    if count > 0:
        fixed = content.replace(OLD, NEW)
        with open(fpath, 'w', encoding='utf-8', newline='') as f:
            f.write(fixed)
        print(f"  Fixed {count:,} paths in {fname}")
    else:
        print(f"  {fname}: already clean")

# ── Step 2: Final verification ────────────────────────────────────────
print()
print("FINAL VERIFICATION")
print("-" * 95)
header = "{:<27} {:>9}  {:<8}  {}".format("File", "Rows", "Status", "Sample face_path")
print(header)
print("-" * 95)

ok_count = 0
broken_count = 0

for fname in csv_files:
    fpath = os.path.join(SPLITS, fname)
    total = sum(1 for _ in open(fpath, encoding='utf-8')) - 1
    with open(fpath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        hdrs = list(reader.fieldnames or [])
        first = next(reader, None)

    path_col = next((c for c in hdrs if 'path' in c.lower()), None)
    if first and path_col:
        sp = first[path_col]
        exists = os.path.exists(sp)
        status = "OK" if exists else "MISSING"
        if exists:
            ok_count += 1
        else:
            broken_count += 1
        truncated = sp[:55] + "..." if len(sp) > 55 else sp
        line = "{:<27} {:>9,}  {:<8}  {}".format(fname, total, status, truncated)
        print(line)
    else:
        print("{:<27} {:>9,}  {:<8}".format(fname, total, "no-path"))

print("-" * 95)
print("  OK: {}   MISSING: {}".format(ok_count, broken_count))
if broken_count == 0:
    print("\n  All CSV paths are correctly resolved!")
else:
    print("\n  WARNING: {} CSV(s) still have missing paths.".format(broken_count))
