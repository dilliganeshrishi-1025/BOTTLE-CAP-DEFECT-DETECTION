import os
import matplotlib.pyplot as plt
from PIL import Image
import shutil

BASE_DIR = '/content/dataset_binary/train'
CLASSES = ['good', 'defective']
NUM_SAMPLES = 3

SAMPLE_OUTPUT_DIR = '/content/ppt_samples'
os.makedirs(SAMPLE_OUTPUT_DIR, exist_ok=True)

fig, axes = plt.subplots(len(CLASSES), NUM_SAMPLES, figsize=(12, 8))

for row, cls in enumerate(CLASSES):
    folder = os.path.join(BASE_DIR, cls)
    files = sorted(os.listdir(folder))[:NUM_SAMPLES]

    for col, fname in enumerate(files):
        fpath = os.path.join(folder, fname)
        img = Image.open(fpath)

        axes[row, col].imshow(img)
        axes[row, col].set_title(f"{cls} - {col+1}")
        axes[row, col].axis('off')

        out_name = f"{cls}_{col+1}.jpg"
        shutil.copy(fpath, os.path.join(SAMPLE_OUTPUT_DIR, out_name))

plt.tight_layout()
plt.savefig('/content/sample_grid.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSample images copied to: {SAMPLE_OUTPUT_DIR}")
print("You can now download them via the Colab file browser (left sidebar),")
print("or right-click 'sample_grid.png' in /content/ to download the combined preview image.")
