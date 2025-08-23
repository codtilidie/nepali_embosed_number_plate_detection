from ultralytics import YOLO
import cv2
import os
import easyocr
from matplotlib import pyplot as plt

# ------------------------------
# CONFIGURATION
# ------------------------------
MODEL_PATH = "C:/Users/ashis/OneDrive/Desktop/nepali_embosed_number_plate_detection/models/best.pt"
IMAGE_PATH = "C:/Users/ashis/OneDrive/Desktop/nepali_embosed_number_plate_detection/resources/IMG_2116.JPG"
OUTPUT_FOLDER = "C:/Users/ashis/OneDrive/Desktop/nepali_embosed_number_plate_detection/output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ------------------------------
# INIT EasyOCR
# ------------------------------
reader = easyocr.Reader(['en'], gpu=False)

# Store detected text and confidence
plate_data = []
conf_score = []

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def show_image(img, title="Image"):
    """Display image with matplotlib"""
    if len(img.shape) == 2:  # grayscale
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(title)
    plt.show()


def recognize_char(cropped_char, reader, idx):
    """OCR on a single character image"""
    # Convert to grayscale
    gray = cv2.cvtColor(cropped_char, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (3,3), 0)

    # Adaptive threshold
    thres = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )

    # Morphological closing (fills gaps)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    cleaned = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)

    # Invert to black chars on white bg
    inverted = cv2.bitwise_not(cleaned)
    
    # Resize for OCR
    resized = cv2.resize(inverted, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)


    # Show thresholded character image
    show_image(resized, f"Thresholded Character {idx}")

    # OCR
    detections = reader.readtext(resized, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

    char_data = []
    char_conf = []

    for (_, text, score) in detections:
        char_data.append(text)
        char_conf.append(score)
 
    
    return char_data, char_conf


def segment_character(img, org_img, class_id):
    """Segment characters from the plate with visualization"""
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Optional sorting for plates
    if class_id == 1:
        centroid_contours = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroid_contours.append((cX, cY, contour))
        sorted_contours = sorted(centroid_contours, key=lambda x: (x[1], x[0]))
        contours = [c[2] for c in sorted_contours]

    # Draw all contours on a copy for visualization
    vis_img = org_img.copy()
    idx = 1
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        plate_area_percentage = (w*h)/(img.shape[0]*img.shape[1])*100

        # Filter likely characters
        if 0.2 < aspect_ratio < 1.0 and 1 < plate_area_percentage < 15:
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Crop character
            char_img = org_img[y:y+h, x:x+w]

            # Recognize character and show thresholded
            chars, scores = recognize_char(char_img, reader, idx)
            plate_data.extend(chars)
            conf_score.extend(scores)

            idx += 1

    show_image(vis_img, "Segmented Characters")


def crop_img(region, img, class_id, padding=15):
    """Crop detected plate and process characters"""
    y1, y2, x1, x2, class_id = region
    h, w = img.shape[:2]

    # Add padding around plate
    y1 = max(0, int(y1 - padding))
    y2 = min(h, int(y2 + padding))
    x1 = max(0, int(x1 - padding))
    x2 = min(w, int(x2 + padding))

    cropped_img = img[y1:y2, x1:x2]
    show_image(cropped_img, "Cropped Plate")

    # Save cropped plate
    save_path = os.path.join(OUTPUT_FOLDER, f"cropped_plate_{len(plate_data)+1}.jpg")
    cv2.imwrite(save_path, cropped_img)

    # Convert to grayscale and threshold for character segmentation
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    thres = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 15, 4)
    show_image(thres, "Thresholded Plate")

    # Segment characters
    segment_character(thres, cropped_img, class_id)


# ------------------------------
# LOAD YOLO MODEL
# ------------------------------
model = YOLO(MODEL_PATH)

# ------------------------------
# LOAD IMAGE
# ------------------------------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

show_image(img, "Input Image")

# ------------------------------
# DETECT PLATES
# ------------------------------
detections = model(img)[0]

for det in detections.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = det
    if score > 0.6:
        region = [y1, y2, x1, x2, int(class_id)]
        crop_img(region, img, int(class_id))

# ------------------------------
# PRINT RESULTS
# ------------------------------
print("Detected Plate Texts:", plate_data)
print("Confidence Scores:", conf_score)
