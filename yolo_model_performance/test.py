import cv2
from ultralytics import YOLO
import math


# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8   # ভালো সাইজ
thickness = 2      # টেক্সট মোটা
box_thickness = 2  # বক্স মোটা

model = YOLO("yolo11n.pt")
video = cv2.VideoCapture("../dataset/Videos/video5.mp4")

cocoClassNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                  "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                  "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                  "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]

while True:
    ret, frame = video.read()

    # if there is no frame the break
    if not ret:
        break
    
    # model
    results = model(frame, conf=0.25, iou=0.7)

    for res in results:
        boxes = res.boxes  
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # draw rectangle bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), box_thickness)

            # Get classname
            classNameInt = int(box.cls[0])
            className = cocoClassNames[classNameInt]

            # Get conf score
            conf = math.ceil(box.conf[0] * 100) / 100
            label = f"{className}: {conf}"

            # --- Text background box ---
            (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), (255, 0, 0), -1)

            # --- Put text on background ---
            cv2.putText(frame, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 

video.release()
cv2.destroyAllWindows()