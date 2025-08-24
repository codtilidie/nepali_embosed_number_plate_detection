# Nepali Embossed Number Plate Detection 🚗📷

This project detects **Nepali embossed vehicle number plates** from **images** and **videos**,  
performs **character segmentation**, and recognizes the license plate using **YOLOv8 + EasyOCR**.

It also saves the cropped plates, segmented characters, and detection results for further processing.

---

## 📂 Project Structure
nepali_embossed_number_plate_detection/
│
├── models/ # YOLO models
│ ├── best.pt # YOLO model for license plates
│ ├── yolov8l.pt # YOLO model for vehicle detection
│
├── resources/ # Sample images and videos
│ ├── IMG_2116.JPG # Sample input image
│ ├── first.mp4 # Sample video
│ ├── segmented_characters/ # Sample segmented character images
│
├── util.py # Helper functions (formatting, CSV writing, contour grouping)
├── detect_image.py # Script to detect plates from an image
├── detect_video.py # Script to detect plates from a video
├── requirements.txt # Python dependencies
└── README.md # This file


---

## ⚙️ Installation

1. Clone the repo:
    ```bash
    git clone https://github.com/aashish/nepali_embossed_number_plate_detection.git
    cd nepali_embossed_number_plate_detection

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Install Git LFS to handle large YOLO models:
   ```bash
   git lfs install

1. Detect from image:
   ```bash
   python detect_image.py

2. Detect from video:
   ```bash
   python detect_video.py

Sample output that canbe helpful 
![Input Image](resources/IMG_2116.JPG)
![Segmented Character 1](resources/segmented_characters/seg_char1.jpg)
![Segmented Character 2](resources/segmented_characters/seg_char2.jpg)
![Video Frame](resources/segmented_characters/frame_105.jpg)
![Cropped Plate](output/cropped_plate_1.jpg)



   
  
