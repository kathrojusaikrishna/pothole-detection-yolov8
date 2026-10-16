# Pothole Detection Using YOLOv8

A computer vision project for detecting and localizing potholes in road images using YOLOv8, trained from scratch without pretrained weights.

## Project Overview

Potholes can cause road accidents, vehicle damage, and unsafe driving conditions. This project uses deep learning and object detection to identify potholes in road images and locate them using bounding boxes.

The YOLOv8n model is initialized with randomly initialized weights and trained on a labeled pothole dataset. Its performance is evaluated on a separate test dataset.

## Objectives

- Detect potholes in road images.
- Localize potholes using bounding boxes.
- Train YOLOv8 from scratch without pretrained weights.
- Evaluate performance using standard object detection metrics.
- Test the trained model on unseen road images.

## Technologies Used

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Matplotlib
- Google Colab
- NVIDIA Tesla T4 GPU

## Model Architecture and Training

This project uses YOLOv8n, the nano variant of YOLOv8.

The model architecture is loaded from its YAML configuration rather than from a pretrained checkpoint.

```python
from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

results = model.train(
    data="/content/pothole_dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    pretrained=False,
    device=0
)
```

Training configuration:

| Parameter          | Value          |
| ------------------ | -------------- |
| Model              | YOLOv8n        |
| Initialization     | Random weights |
| Pretrained weights | Not used       |
| Image size         | 640 × 640      |
| Maximum epochs     | 100            |
| Batch size         | 8              |

The actual number of epochs may be lower if early stopping is triggered.

## Dataset

The project uses a pothole detection dataset with images and bounding-box annotations in YOLO format.

The dataset is organized into training, validation, and test splits. Dataset files are not included in this repository.

The reported test evaluation contains 67 images and 154 annotated pothole instances.

## Evaluation Results

The trained model was evaluated on the test dataset.

| Metric    | Result |
| --------- | -----: |
| Precision | 67.15% |
| Recall    | 69.03% |
| mAP@50    | 70.54% |
| mAP@50–95 | 43.20% |

These metrics represent the results from the evaluated model checkpoint and test run.

## Sample Prediction

The model predicts pothole locations using bounding boxes and confidence scores.

Add a prediction image to the repository at `results/pothole_detection.png` to display it here.

```markdown
![Pothole Detection Result](results/pothole_detection.png)
```

## Repository Structure

```text
pothole-detection-yolov8/
├── notebooks/
│   └── Pathole_detection_scratch.ipynb
├── results/
│   └── pothole_detection.png
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd pothole-detection-yolov8
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the URL of your GitHub repository.

### 2. Install dependencies

Create and activate a virtual environment, then install the required packages.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Prepare the dataset

Download the pothole dataset and arrange it in YOLO format. Update the paths in `data.yaml` to match your dataset directory.

### 4. Train the model

Open `notebooks/Pathole_detection_scratch.ipynb` in Google Colab and run the setup, dataset preparation, and training cells.

Ensure that the model is initialized using `YOLO("yolov8n.yaml")` and that training uses `pretrained=False`.

### 5. Run inference

After training, load the generated `best.pt` checkpoint.

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="path/to/road_image.jpg",
    conf=0.25,
    save=True
)
```

Replace the image path with the path to your road image.

## Model Weights

The trained `best.pt` checkpoint is generated during training and is not included in this repository by default. It can be shared separately or through a model-hosting service.

## Future Improvements

- Improve detection of small and partially visible potholes.
- Experiment with image augmentation techniques.
- Evaluate the effect of longer training and different model configurations.
- Test on road images captured under different lighting and weather conditions.
- Explore real-time pothole detection using video streams.

## Author

Saikrishna

## License

Add a license after checking the licensing terms of the dataset and dependencies.
