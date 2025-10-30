# Dataset Directory

This directory contains the training data and configuration files for the Equipment Verification System.

## Directory Structure

```
data/
├── images/                 # Training images
│   ├── train/             # Training set images
│   ├── val/               # Validation set images
│   └── test/              # Test set images
├── labels/                # YOLO format labels
│   ├── train/             # Training set labels
│   ├── val/               # Validation set labels
│   └── test/              # Test set labels
├── checklists/            # Equipment checklists
│   ├── construction.json  # Construction safety checklist
│   ├── medical.json       # Medical equipment checklist
│   └── laboratory.json    # Laboratory safety checklist
└── README.md              # This file
```

## Dataset Preparation

### 1. Image Collection

Collect images of people wearing various equipment items. Ensure diversity in:
- Lighting conditions
- Angles and poses
- Backgrounds
- Equipment variations
- Different people

### 2. Labeling

Use YOLO format for bounding box annotations:

```
class_id x_center y_center width height
```

Where all coordinates are normalized (0-1).

### 3. Class Mapping

Default class IDs:
- 0: hard_hat
- 1: safety_vest
- 2: safety_boots
- 3: safety_glasses
- 4: gloves
- 5: ear_protection
- 6: respirator
- 7: scrubs
- 8: stethoscope
- 9: face_mask
- 10: lab_coat
- 11: safety_goggles
- 12: apron
- 13: shoe_covers
- 14: hair_net
- 15: safety_shoes

### 4. Data Splitting

The training script will automatically split your data into train/validation/test sets (80%/10%/10% by default).

## Sample Data

For testing purposes, you can use any images with people and equipment. The system will work with the pretrained YOLOv8 model even without custom training data.

## Adding New Equipment

1. Add new class names to the configuration
2. Collect and label training data
3. Retrain the model
4. Update equipment checklists

See `TRAINING_GUIDE.md` for detailed instructions.


