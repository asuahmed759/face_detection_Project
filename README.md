# Face Detection Project

A deep learning face detection model built with TensorFlow/Keras and a VGG16 backbone. The model takes a 120x120 image and predicts both **whether a face is present** and **the bounding box coordinates** of that face. Includes a real-time webcam detection demo using OpenCV.

## Overview

This project walks through the full pipeline of building a custom object detector from scratch:

1. Collecting and labeling image data
2. Augmenting data with Albumentations
3. Building a multi-output CNN (classification + bounding box regression) on top of VGG16
4. Training with a custom Keras model class and custom loss functions
5. Running real-time face detection on a live webcam feed

## Project Structure

```
.
├── data/                  # Raw collected images + LabelMe annotations (train/test/val split)
├── aug_data/              # Augmented images + labels generated from data/ (gitignored)
├── logs/                  # TensorBoard training logs (gitignored)
├── facetracker.h5         # Trained model weights
└── face_detection_project.ipynb   # Main notebook — the entire pipeline
```

## Pipeline

### 1. Data Collection & Annotation
- Images captured via webcam using OpenCV
- Each image manually annotated with a bounding box using [LabelMe](https://github.com/wkentaro/labelme)

### 2. Data Pipeline & Augmentation
- Images loaded into a `tf.data.Dataset` pipeline
- Data split into train / test / validation sets
- [Albumentations](https://albumentations.ai/) used to augment each image 60x (random crop, flips, brightness/contrast, gamma, RGB shift) to expand the training set and improve generalization

### 3. Model Architecture
A functional-API Keras model with a shared **VGG16** (ImageNet weights, `include_top=False`) backbone and two output heads:

- **Classification head** — `GlobalMaxPooling2D → Dense(2048, relu) → Dense(1, sigmoid)` → face / no-face confidence
- **Regression head** — `GlobalMaxPooling2D → Dense(2048, relu) → Dense(4, sigmoid)` → bounding box coordinates `[xmin, ymin, xmax, ymax]`, normalized to `[0,1]`

### 4. Custom Training Loop
A custom `FaceTracker` model class (subclassing `tf.keras.Model`) overrides `train_step` and `test_step` to combine:
- **Classification loss** — Binary Crossentropy
- **Localization loss** — a custom function penalizing both corner-point distance and width/height differences

```
total_loss = localization_loss + 0.5 * classification_loss
```

### 5. Real-Time Detection
The trained model is loaded and run on a live webcam feed (OpenCV `VideoCapture`), drawing a bounding box and label around any detected face in real time.

## Requirements

- Python 3.10
- TensorFlow / Keras 3
- OpenCV (`opencv-python`)
- Albumentations
- LabelMe (for annotation)
- NumPy, Matplotlib

Install dependencies:
```bash
pip install tensorflow opencv-python albumentations labelme numpy matplotlib
```

## Usage

1. Open `face_detection_project.ipynb` in Jupyter / VS Code
2. Run the cells sequentially from top to bottom:
   - Sections 1–7: data collection, loading, and augmentation (skip if `data/` and `aug_data/` already exist)
   - Section 8: build the model
   - Section 9–10: compile and train the model
   - Section 11: run predictions on the test set, save the model, and launch real-time webcam detection

To re-run only inference (skip training), load the saved model directly:
```python
from tensorflow.keras.models import load_model
facetracker = load_model('facetracker.h5')
```

## Notes

- The model was trained on a small custom dataset (~90 source images, augmented 60x). Detection accuracy depends heavily on how similar your webcam conditions are to the training data (lighting, background, distance from camera).
- The real-time detection cell auto-adapts the bounding-box scale to your webcam's actual frame size, so it works regardless of resolution.
