import os
import shutil

# Class ID -> binary label mapping (from your data.yaml)
# 0: Broken Cap, 1: Broken Ring, 2: Good Cap, 3: Loose Cap, 4: No Cap
CLASS_MAP = {
    0: 'defective',  # Broken Cap
    1: 'defective',  # Broken Ring
    2: 'good',        # Good Cap
    3: 'defective',  # Loose Cap
    4: 'defective',  # No Cap
}

# Change this if your dataset folder is named differently
BASE_DIR = '/content/dataset'
OUTPUT_DIR = '/content/dataset_binary'
SPLITS = ['train', 'valid', 'test']

counts = {}

for split in SPLITS:
    images_dir = os.path.join(BASE_DIR, split, 'images')
    labels_dir = os.path.join(BASE_DIR, split, 'labels')

    if not os.path.isdir(images_dir):
        print(f"Skipping {split}: folder not found at {images_dir}")
        continue

    counts[split] = {'good': 0, 'defective': 0}

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue

        label_path = os.path.join(labels_dir, label_file)
        image_name = label_file.replace('.txt', '.jpg')
        image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            image_name_png = label_file.replace('.txt', '.png')
            image_path_png = os.path.join(images_dir, image_name_png)
            if os.path.exists(image_path_png):
                image_path = image_path_png
                image_name = image_name_png
            else:
                print(f"Warning: no matching image for {label_file}")
                continue

        with open(label_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            continue

        class_ids = [int(line.split()[0]) for line in lines if line.strip()]

        if any(CLASS_MAP[cid] == 'defective' for cid in class_ids):
            label = 'defective'
        else:
            label = 'good'

        dest_dir = os.path.join(OUTPUT_DIR, split, label)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(image_path, os.path.join(dest_dir, image_name))

        counts[split][label] += 1

print("\n=== Class counts per split ===")
for split, c in counts.items():
    total = c['good'] + c['defective']
    print(f"{split}: good={c['good']}, defective={c['defective']}, total={total}")
