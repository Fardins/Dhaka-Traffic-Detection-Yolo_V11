# Dhaka Traffic Detection using YOLO-V11

<p align="center">
  <img src="./outputs/input-2_output.gif" alt="Cat Output" width="45%" />
  <img src="./outputs/input-3_output.gif" alt="Dog Output" width="45%" />
</p>
<p align="center">
  <img src="./outputs/output.gif" alt="Cat Output" width="45%" />
</p>

Streamlit App Link: https://dhaka-traffic-detection-yolov11-atick.streamlit.app/

## Overview

Dhaka Traffic Detection using YOLOv11 is a computer vision project designed to identify and classify various vehicles in traffic videos. It leverages the state-of-the-art YOLOv11 model for real-time object detection and is tailored to recognize 21 different vehicle categories commonly found in Dhaka's traffic.

## Features

- **Custom Vehicle Detection:** Detects and classifies 21 vehicle types including cars, buses, motorcycles, rickshaws, and more.

- **Real-Time Processing:** Uses the YOLOv11 model for efficient and real-time video processing.

- **Interactive Streamlit App:** Allows users to upload videos or select sample videos for analysis.

- **Post-Processing with FFmpeg:** Ensures high-quality output videos with minimal file size.

- **Pre-Trained Weights:** Trained on a dataset specifically prepared for Dhaka traffic conditions.

## Folder Structure

```bash
Dhaka-Traffic-Detection-Yolo_V11
|── data
|    ── train  # Training images and labels
|    ── val    # Validation images and labels
|── inputs
|    ── input-1.mp4
|    ── input-2.mp4
|── outputs
|── runs
|    ── detect
|         ── train
|              ── weights
|                   ── best.pt
|── streamlit_output
|── config.yaml
|── requirements.txt
|── train.py
|── predict.py
|── app.py
|── yolo11n.pt
```
## Project Structure
```bash
Dhaka-Traffic-Detection-Yolo_V11
|── data
|    ├── train               # Training images and labels
|    ├── val                 # Validation images and labels
|
|── inputs                   # Videos for prediction
|    ├── input-1.mp4
|    ├── input-2.mp4
|
|── outputs                  # Folder for saving predicted outputs
|    ├── (processed videos will be saved here)
|
|── runs                     # YOLO training outputs
|    ├── detect
|         ├── train
|              ├── weights
|                   ├── best.pt
|
|── streamlit_output         # Streamlit app output videos
|    ├── (processed videos will be saved here)
|
|── config.yaml              # Dataset configuration file
|── requirements.txt         # Project dependencies
|── train.py                 # Script for training the YOLOv11 model
|── predict.py               # Script for running predictions on input videos
|── app.py                   # Streamlit app script
|── yolo11n.pt               # Pre-trained YOLOv11 model weights


```
## Number Of Classes Detected
1. Ambulance
2. Auto Rickshaw
3. Bicycle
4. Bus
5. Car
6. Garbage Van
7. Human Hauler
8. Minibus
9. Minivan
10. Motorbike
11. Pickup
12. Army Vehicle
13. Police Car
14. Rickshaw
15. Scooter
16. SUV
17. Taxi
18. Three Wheelers (CNG)
19. Truck
20. Van
21. Wheelbarrow

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Fardins/Dhaka-Traffic-Detection-Yolo_V11.git
    cd Dhaka-Traffic-Detection-Yolo_V11
    ```
2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model
Train the YOLOv11 model on your custom dataset:
```bash
python train.py
```

### Predicting on Videos
Run predictions on videos from the inputs folder:
```bash
python predict.py
```

### Streamlit App
Launch the interactive web application:
```bash
streamlit run app.py
```
Upload my traffic video or select a sample video for object detection.
My app will be available locally at: http://localhost:8501.
Streamlit App: https://dhaka-traffic-detection-yolov11-atick.streamlit.app/

## Configuration

The config.yaml file defines the dataset structure and class names. Ensure the paths to training and validation data are correctly set:
```bash
train: images/train  # Training images
val: images/val      # Validation images
nc: 21               # Number of classes
names: ["ambulance", "auto rickshaw", ..., "wheelbarrow"]
```

## Results

### Sample Outputs

Processed videos are saved in the `outputs` and `streamlit_output` folders. Below is an example of detected traffic in Dhaka:

- Input: `inputs/input-1.mp4`
- Output: `outputs/input-1_output.mp4`

## Output Example

After processing a video, the output will be an annotated video file showing detected objects with bounding boxes and labels. The output video is displayed in the Streamlit app or saved in the `output/` folder for batch processing.

Here are some examples of videos and the resulting objects generated by the trained model:
### Example 1:
- **Video Input and Output Objects Detection:**
<p align="center">
  <img src="./inputs/input.gif" alt="Input" width="45%" />
  <img src="./outputs/output.gif" alt="Output" width="45%" />
</p>

### Example 2:
- **Video Input and Output Objects Detection:**
<p align="center">
  <img src="./inputs/input-2.gif" alt="Input" width="45%" />
  <img src="./outputs/input-2_output.gif" alt="Output" width="45%" />
</p>

### Example 3:
- **Video Input and Output Objects Detection:**
<p align="center">
  <img src="./inputs/input-3.gif" alt="Input" width="45%" />
  <img src="./outputs/input-3_output.gif" alt="Output" width="45%" />
</p>

### Example 4:
- **Video Input and Output Objects Detection:**
<p align="center">
  <img src="./inputs/input-4.gif" alt="Input" width="45%" />
  <img src="./outputs/input-4_output.gif" alt="Output" width="45%" />
</p>

### Example 5:
- **Video Input and Output Objects Detection:**
<p align="center">
  <img src="./inputs/input-1.gif" alt="Input" width="45%" />
  <img src="./outputs/input-1_output.gif" alt="Output" width="45%" />
</p>

You can view these examples within the Streamlit app or try it yourself by selecting an image from the dropdown and running segmentation.

## Requirements

- Python 3.8+
- Dependencies (see `requirements.txt`):
  - opencv-python
  - pillow
  - ultralytics
  - streamlit
  - moviepy
  - imageio[ffmpeg]

## Future Enhancements

- Instance Segmentation: Extend to segment individual objects.
- Edge Device Compatibility: Optimize for deployment on mobile devices or edge hardware.
- Dashboard Integration: Add visualization features for traffic analytics.

## Acknowledgments
- YOLOv11 by Ultralytics
- OpenCV for video processing
- Streamlit for interactive app development
- FFmpeg for efficient video post-processing

## Contact
For questions or collaboration, please contact:
- **Md Atickur Rahman**: atickft13129@gmail.com
- **GitHub**: [GitHub](https://github.com/Fardins)
