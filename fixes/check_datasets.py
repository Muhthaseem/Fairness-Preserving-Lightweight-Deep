"""
Full dataset audit: raw folder structure + split CSV path resolution + face counts.
"""
import csv
import os
import glob

SPLITS = r'D:\M3\DeepFake_Research\outputs\splits'
FACES  = r'D:\M3\DeepFake_Research\outputs\faces'

# ── 1. Dataset folders ──────────────────────────────────────────────────
print('=' * 65)
print('DATASET FOLDER AUDIT')
print('=' * 65)

checks = [
    ('FF++/original',          r'D:\M3\DeepFake_Research\Datasets\FF++\FaceForensics++_C23\original'),
    ('FF++/Deepfakes',         r'D:\M3\DeepFake_Research\Datasets\FF++\FaceForensics++_C23\Deepfakes'),
    ('FF++/Face2Face',         r'D:\M3\DeepFake_Research\Datasets\FF++\FaceForensics++_C23\Face2Face'),
    ('FF++/FaceSwap',          r'D:\M3\DeepFake_Research\Datasets\FF++\FaceForensics++_C23\FaceSwap'),
    ('FF++/NeuralTextures',    r'D:\M3\DeepFake_Research\Datasets\FF++\FaceForensics++_C23\NeuralTextures'),
    ('FF++/FaceShifter',       r'D:\M3\DeepFake_Research\Datasets\FF++\FaceForensics++_C23\FaceShifter'),
    ('FF++/DeepFakeDetection', r'D:\M3\DeepFake_Research\Datasets\FF++\FaceForensics++_C23\DeepFakeDetection'),
    ('Celeb-DF/Celeb-real',    r'D:\M3\DeepFake_Research\Datasets\Celeb-DF\Celeb-real'),
    ('Celeb-DF/Celeb-synthesis', r'D:\M3\DeepFake_Research\Datasets\Celeb-DF\Celeb-synthesis'),
    ('Celeb-DF/YouTube-real',  r'D:\M3\DeepFake_Research\Datasets\Celeb-DF\YouTube-real'),
    ('Celeb-DF/List_test.txt', r'D:\M3\DeepFake_Research\Datasets\Celeb-DF\List_of_testing_videos.txt'),
    ('DFD/DFD_manipulated',    r'D:\M3\DeepFake_Research\Datasets\DFD\DFD_manipulated_sequences'),
    ('DFD/DFD_original',       r'D:\M3\DeepFake_Research\Datasets\DFD\DFD_original sequences'),
]

for label, path in checks:
    exists = os.path.exists(path)
    status = 'OK' if exists else 'MISSING'
    extra = ''
    if exists and os.path.isdir(path):
        n = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        extra = '({} files)'.format(n)
    print('  [{}]  {:30s}  {}'.format(status, label, extra))

# ── 2. Split CSV audit ──────────────────────────────────────────────────
print()
print('=' * 65)
print('SPLIT CSV AUDIT')
print('=' * 65)

ok_count = 0
broken_count = 0
csv_files = sorted(f for f in os.listdir(SPLITS) if f.endswith('.csv'))

for fname in csv_files:
    fpath = os.path.join(SPLITS, fname)
    total = sum(1 for _ in open(fpath, encoding='utf-8-sig')) - 1
    with open(fpath, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        hdrs = list(reader.fieldnames or [])
        first = next(reader, None)

    path_col = next((c for c in hdrs if 'path' in c.lower()), None)
    if first and path_col:
        sp = first[path_col]
        exists = os.path.exists(sp)
        if exists:
            ok_count += 1
            tag = 'OK'
        else:
            broken_count += 1
            tag = 'MISSING'
        size_mb = os.path.getsize(fpath) / 1e6
        print('  [{}]  {:25s}  {:>9,} rows  {:5.1f} MB'.format(
            tag, fname, total, size_mb))
        if not exists:
            print('       broken path: {}'.format(sp[:70]))
    else:
        print('  [??]  {:25s}  {:>9,} rows  (no path column)'.format(fname, total))

print()
print('  CSVs OK={}, MISSING={}'.format(ok_count, broken_count))

# ── 3. Face image count ─────────────────────────────────────────────────
print()
print('=' * 65)
print('FACE IMAGES')
print('=' * 65)
face_count = sum(1 for _ in glob.iglob(FACES + '/**/*.jpg', recursive=True))
print('  Total face JPGs in outputs/faces/: {:,}'.format(face_count))

# ── 4. Model checkpoints ────────────────────────────────────────────────
MODELS = r'D:\M3\DeepFake_Research\outputs\models'
print()
print('=' * 65)
print('MODEL CHECKPOINTS')
print('=' * 65)
for fname in sorted(os.listdir(MODELS)):
    fp = os.path.join(MODELS, fname)
    if os.path.isfile(fp):
        size_mb = os.path.getsize(fp) / 1e6
        print('  {:40s}  {:6.1f} MB'.format(fname, size_mb))
    elif os.path.isdir(fp):
        n = len(os.listdir(fp))
        print('  {:40s}  [{} files]'.format(fname + '/', n))

print()
print('Audit complete.')
