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
📸 Sample Outputs
🔹 Input Example

Here is a sample input image:

<img src="resources/IMG_2116.JPG" alt="Input Image" width="400"/>
🔹 Detected Characters from Photo

Below are the segmented characters detected from an image:

<p float="left"> <img src="images/for_photo/character_1.png" alt="Character 1" width="100"/> <img src="images/for_photo/character_2.png" alt="Character 2" width="100"/> <img src="images/for_photo/character_3.png" alt="Character 3" width="100"/> <img src="images/for_photo/character_4.png" alt="Character 4" width="100"/> <img src="images/for_photo/character_5.png" alt="Character 5" width="100"/> <img src="images/for_photo/character_6.png" alt="Character 6" width="100"/> <img src="images/for_photo/character_7.png" alt="Character 7" width="100"/> <img src="images/for_photo/detected_character_contoured.png" alt="Final Contour" width="150"/> </p>
🎥 Detected Plates from Video

Here are some frames detected from a video input:

<p float="left"> <img src="images/for_video/figure10.png" alt="Frame 1" width="200"/> <img src="images/for_video/Figure12.png" alt="Frame 2" width="200"/> <img src="images/for_video/figure11.png" alt="Frame 4" width="200"/> <img src="images/for_video/Figure13.png" alt="Frame 3" width="200"/> <img src="images/for_video/Figure14.png" alt="Frame 4" width="200"/> <img src="images/for_video/Figure15.png" alt="Frame 4" width="200"/> <img src="images/for_video/Figure16.png" alt="Frame 4" width="200"/> <img src="images/for_video/Figure17.png" alt="Frame 4" width="200"/> <img src="images/for_video/Figure18.png" alt="Frame 4" width="200"/> </p></p></p>
![Detected 8](images/for_photo/detected_character_contoured.png)   
  
