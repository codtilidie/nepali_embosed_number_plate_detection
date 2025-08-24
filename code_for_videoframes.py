from ultralytics import YOLO
import cv2
import easyocr
from util import get_car, is_valid_plate, format_license, write_csv, group_contours_by_row
from sort.sort import*
from matplotlib import pyplot as plt

reader = easyocr.Reader(['en'], gpu = False)
detection = []
plate_data = []
conf_score = []
results = {}
mot_tracker = Sort()

def show_image(img, title="Image"):
    """Display image with matplotlib"""
    if len(img.shape) == 2:  # grayscale
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(title)
    plt.show()



def recognize_char(char_region, img): 
    
    cropped_char = img[int(char_region[0]):int(char_region[1]), int(char_region[2]):int(char_region[3])]

    # Define padding size
    padding = 10

    # Get the dimensions of the character image
    height, width = cropped_char.shape

    # Create a new image with padding
    padded_image = np.ones((height + 2*padding, width + 2*padding), dtype=np.uint8)

    # Place the character image in the center of the padded image
    padded_image[padding:padding+height, padding:padding+width] = cropped_char
    print("in recognize_char function")
    show_image(padded_image,"padded_image")
    
    detections= reader.readtext(padded_image)  
    for detection in detections:
        bbox, text, score = detection
        plate_data.append(text)
        conf_score.append(score)
        
def segmentcharacter(img, org_img, class_id):
    print("in segmentcharacter")
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    show_image(img,"segmented_image")
    sorted_contours = []  # Always initialize

    if class_id == 1 and len(contours) > 0:
        centroid_contours = []

        # Calculate centroids
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroid_contours.append((cX, cY, contour))

        if len(centroid_contours) > 0:
            rows = group_contours_by_row(centroid_contours)
            # Flatten sorted rows
            sorted_contours = [contour for row in rows for contour in sorted(row, key=lambda x: x[0])]
                
    # Loop safely
    for _, _, cnt in sorted_contours:
        plate = img.shape
        (w, h) = cv2.boundingRect(cnt)[2:]
        plate_char_percentage = ((w * h)/(plate[0] * plate[1]))*100
        aspect_ratio = w / float(h)
        
        if 0.4 < aspect_ratio < 0.8 and 2 < plate_char_percentage < 8:
            x1, y1, w, h = cv2.boundingRect(cnt)
            char_region = [y1, y1+h, x1, x1+w]
            recognize_char(char_region, img)


def filter_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thres = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 8)
    # cv2.imshow('', thres)
    # cv2.waitKey(0)
    
    # Apply Bilateral Filter
    bilateral_filtered = cv2.bilateralFilter(gray, 5,150,150)
    
    # Define the kernel for morphological operations
    kernel_size = (2 ,2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # Example of morphological closing with adjusted kernel
    thres_adaptive_closed = cv2.morphologyEx(bilateral_filtered, cv2.MORPH_CLOSE, kernel)
    

    # Apply CLAHE to enhance local contrast
    # img_clahe = cv2.equalizeHist(thres_adaptive_closed)
    # Create a CLAHE object (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(3 ,3))
    
    # Apply CLAHE to enhance local contrast
    img_clahe = clahe.apply(thres_adaptive_closed)

        # cv2.imshow('clahe', img_clahe)   
    
    thres = cv2.adaptiveThreshold(img_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 30)
    # cv2.imshow('final', thres)
    # cv2.waitKey(0)
    show_image(thres,"threshold image")
        
    segmentcharacter(thres, img, region[4])


def crop_img(region, img):
    cropped_img = img[int(region[0]):int(region[1]), int(region[2]):int(region[3])]
        # cv2.imshow('', cropped_img)
        # cv2.waitKey(0)
    
    scale_percent = 200  # percent of original size
    width = int(cropped_img.shape[1] * scale_percent / 100)
    height = int(cropped_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    cropped_img = cv2.resize(cropped_img, dim, interpolation=cv2.INTER_AREA)
    show_image(cropped_img,"cropped_img")
        
    # Convert the image to the YCrCb color space
    ycrcb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2YCrCb)
    # Split the channels
    y, cr, cb = cv2.split(ycrcb)
    # Apply Histogram Equalization to the Y channel
    y_eq = cv2.equalizeHist(y)
    # Merge the channels back
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    # Convert back to BGR color space
    image_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    # Apply CLAHE to the Y channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y)
    # Merge the CLAHE enhanced Y channel back with Cr and Cb channels
    ycrcb_clahe = cv2.merge([y_clahe, cr, cb])
    # Convert back to BGR color space
    cropped_img = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2BGR)
    # cv2.imshow("histo", cropped_img)
    # cv2.waitKey(0)
    
    #sharpnening the image
    # Apply Gaussian Blur to the image
    blurred = cv2.GaussianBlur(cropped_img, (9, 9), 10.0)
    # Apply the unsharp mask
    cropped_img = cv2.addWeighted(cropped_img, 1.5, blurred, -0.5, 0)
    # cv2.imshow("sharp", cropped_img)
    # cv2.waitKey(0)

    # cv2.imwrite('img_3.jpg',cropped_img)
        # cv2.imshow('', cropped_img)
        # cv2.waitKey(0)
    
    
    filter_img(cropped_img)

vehicles = [2, 3, 5, 7]

#further development
license_plate_detector = YOLO("C:/Users/ashis/OneDrive/Desktop/nepali_embosed_number_plate_detection/models/best.pt")
coco_model = YOLO("C:/Users/ashis/OneDrive/Desktop/nepali_number_embosed_plate_detection/models/yolov8l.pt")
cap = cv2.VideoCapture('C:/Users/ashis/OneDrive/Desktop/nepali_embosed_number_plate_detection/resources/first.mp4')

#reading the frames from the video
frame_nmr = -1
ret = True

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        break

    if 100 < frame_nmr < 120:
        results[frame_nmr] = {}

        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if class_id in vehicles:
                detections_.append([x1, y1, x2, y2])

        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Match plate with car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            # Draw car box (blue)
            if car_id != -1:
                cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (255, 0, 0), 2)
                cv2.putText(frame, f'Car {car_id}', (int(xcar1), int(ycar1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Draw license plate box (red)
            if score > 0.6:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, f'Plate {score:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if car_id != -1:
                    region = [y1, y2, x1, x2, int(class_id)]
                    crop_img(region, frame)

                    print(plate_data)
                    print(is_valid_plate(plate_data))

                    if is_valid_plate(plate_data):
                        license_plate_data = format_license(plate_data)
                        print(license_plate_data)

                        if license_plate_data is not None:
                            results[frame_nmr][car_id] = {
                                'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                'license_plate': {
                                    'bbox': [x1, y1, x2, y2],
                                    'text': license_plate_data,
                                    'bbox_score': score,
                                    'text_score': sum(conf_score) / max(1, len(conf_score))
                                }
                            }
                            write_csv(results, './results.csv')

                    plate_data.clear()
                    conf_score.clear()

        # Show frame with debug boxes
        show_image(frame, f"Frame {frame_nmr}")


print("Result was successfully saved in the results.csv.")

