# BOTTLE-CAP-DEFECT-DETECTION
CNN BASED BOTTLE CAP DEFECT DETECTION 
## Problem Statement
Automated detection of defective bottle caps (broken cap, broken ring, loose cap, missing cap) on a beverage bottling line using CNN-based visual inspection, to reduce leakage, contamination, and manual inspection errors.

## Dataset
- Source: [Bottle Cap Defect Dataset (Roboflow Universe)](https://universe.roboflow.com/project-jxzvy/bottle-cap-iuzcs)
- 1,432 images (after augmentation), reorganized into binary classification: Good vs Defective
- Split: Train (1202) / Validation (116) / Test (114)

## Model
- MobileNetV2 (transfer learning, ImageNet pretrained)
- Binary classification head: GlobalAveragePooling → Dense(128) → Dropout → Dense(1, sigmoid)
- Class imbalance handled via class-weighted loss

## Results (Test Set)
- Defect Detection Rate (Recall): 80.9%
- False Positive Rate: 8.0%
- Precision: 97.3%

## Project Structure

## Team
- Dilli Ganesh B (231801033)
- Hemakumar U (231801054)

Supervisor: Savithri S
