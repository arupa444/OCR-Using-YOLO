# Performing Object Detection Using YOLO11n

This project demonstrates **object detection** using the YOLO11n model. The notebook walks through the process of loading the YOLO model, performing inference on images/videos, and visualizing the predictions.

---

## 🚀 Features

* Load and configure **YOLO11n** pre-trained model.
* Perform object detection on:

  * Images
  * Videos
  * Webcam stream
* Visualize bounding boxes and class labels.
* Export detection results.

---

## 📂 Project Structure

```
├── performingObjectDetectionUsingYOLO11n.ipynb   # Main Jupyter Notebook
├── you can use COLOB                             # Dependencies
├── README.md                                     # Project Documentation
└── runs/                                         # YOLO output results (images/videos with predictions)
```

---

## ⚙️ Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/YOLO11n-ObjectDetection.git
   cd YOLO11n-ObjectDetection
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Mac/Linux
   .venv\Scripts\activate      # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Usage

### 1. Run the Jupyter Notebook

```bash
jupyter notebook performingObjectDetectionUsingYOLO11n.ipynb
```

### 2. Detect Objects on Images

Inside the notebook:

```python
!yolo predict model=yolo11n.pt source='image.jpg'
```

### 3. Detect Objects on Video

```python
!yolo predict model=yolo11n.pt source='video.mp4'
```

### 4. Live Webcam Detection

```python
!yolo predict model=yolo11n.pt source=0
```

---

## 📦 Output

* Detected images and videos are saved in:

  ```
  runs/detect/
  ```
* Each run creates a new folder with the timestamp.

---

## 📌 Requirements

* Python 3.8+
* Jupyter Notebook
* Ultralytics YOLO
* OpenCV
* Torch

Install directly with:

```bash
pip install ultralytics opencv-python torch torchvision
```

---

## 🛠 Example

Example of running detection on an image:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model("image.jpg")
results.show()
```

---

## 📄 License

This project is licensed under the MIT License - feel free to use and modify.
