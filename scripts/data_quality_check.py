import os
from PIL import Image
import hashlib

BASE_DIR = '/content/dataset_binary'
SPLITS = ['train', 'valid', 'test']
CLASSES = ['good', 'defective']

corrupt_files = []
resolutions = {}
hashes = {}
duplicates = []

total_checked = 0

for split in SPLITS:
    for cls in CLASSES:
        folder = os.path.join(BASE_DIR, split, cls)
        if not os.path.isdir(folder):
            continue

        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            total_checked += 1

            try:
                img = Image.open(fpath)
                img.verify()
                img = Image.open(fpath)
                w, h = img.size
            except Exception as e:
                corrupt_files.append(fpath)
                continue

            res = f"{w}x{h}"
            resolutions[res] = resolutions.get(res, 0) + 1

            with open(fpath, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash in hashes:
                duplicates.append((fpath, hashes[file_hash]))
            else:
                hashes[file_hash] = fpath

print(f"=== Data Quality Report ===")
print(f"Total images checked: {total_checked}")
print(f"\nCorrupt/unreadable images: {len(corrupt_files)}")
for f in corrupt_files[:10]:
    print(f"  - {f}")

print(f"\nResolution breakdown (top 10 most common):")
sorted_res = sorted(resolutions.items(), key=lambda x: -x[1])
for res, count in sorted_res[:10]:
    print(f"  {res}: {count} images")
print(f"  Total unique resolutions: {len(resolutions)}")

print(f"\nExact duplicate images found: {len(duplicates)}")
for f1, f2 in duplicates[:10]:
    print(f"  - {f1}  ==  {f2}")
