"""
Bottle Cap Defect Detection - Deployment Inference Script
============================================================
This script represents the core logic that would run on an edge device
(e.g. Raspberry Pi 4, NVIDIA Jetson Nano) mounted at the capping station
on a bottling line.

Production flow:
  1. Camera captures an image of each bottle cap as it passes.
  2. This script loads the trained model once at startup (not per-image).
  3. For each incoming image, it runs inference and returns Good/Defective.
  4. A "Defective" result would trigger a reject-arm / conveyor diverter
     signal (simulated here as a print statement / log entry).

Usage:
  python deploy_inference.py --image path/to/cap_image.jpg
  python deploy_inference.py --folder path/to/images/   (batch mode)
"""

import argparse
import os
import time
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "bottle_cap_model.h5"
CONFIDENCE_THRESHOLD = 0.5  # probability above this = Defective
IMG_SIZE = (224, 224)


class CapInspector:
    """Loads the model once and runs repeated inference — mirrors how this
    would run continuously on a line, not reloading the model per image."""

    def __init__(self, model_path=MODEL_PATH):
        print(f"[INIT] Loading model from {model_path} ...")
        start = time.time()
        self.model = load_model(model_path)
        print(f"[INIT] Model loaded in {time.time() - start:.2f}s. Ready for inference.")

    def predict(self, image_path):
        """Run inference on a single image. Returns (label, confidence, raw_prob)."""
        img = keras_image.load_img(image_path, target_size=IMG_SIZE)
        img_array = keras_image.img_to_array(img)
        img_array = preprocess_input(np.expand_dims(img_array, axis=0))

        raw_prob = float(self.model.predict(img_array, verbose=0)[0][0])
        is_defective = raw_prob > CONFIDENCE_THRESHOLD
        label = "DEFECTIVE" if is_defective else "GOOD"
        confidence = raw_prob if is_defective else 1 - raw_prob

        return label, confidence, raw_prob

    def inspect_and_act(self, image_path):
        """Runs prediction and simulates the downstream production action."""
        start = time.time()
        label, confidence, raw_prob = self.predict(image_path)
        latency_ms = (time.time() - start) * 1000

        filename = os.path.basename(image_path)
        print(f"[{filename}] -> {label} (confidence: {confidence:.1%}, "
              f"latency: {latency_ms:.1f}ms)")

        if label == "DEFECTIVE":
            print(f"    >>> ACTION: Trigger reject-arm for {filename}")
        else:
            print(f"    >>> ACTION: Pass through to packaging")

        return label, confidence


def main():
    parser = argparse.ArgumentParser(description="Bottle cap defect inspection")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--folder", type=str, help="Path to a folder of images (batch mode)")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to the .h5 model file")
    args = parser.parse_args()

    inspector = CapInspector(model_path=args.model)

    if args.image:
        inspector.inspect_and_act(args.image)

    elif args.folder:
        valid_ext = (".jpg", ".jpeg", ".png")
        images = [f for f in os.listdir(args.folder) if f.lower().endswith(valid_ext)]
        if not images:
            print(f"No images found in {args.folder}")
            return

        print(f"\n[BATCH MODE] Processing {len(images)} images...\n")
        results = {"GOOD": 0, "DEFECTIVE": 0}
        for fname in sorted(images):
            label, _ = inspector.inspect_and_act(os.path.join(args.folder, fname))
            results[label] += 1

        print(f"\n[SUMMARY] Good: {results['GOOD']}, Defective: {results['DEFECTIVE']}, "
              f"Total: {len(images)}")

    else:
        print("Please provide --image <path> or --folder <path>. Use -h for help.")


if __name__ == "__main__":
    main()
