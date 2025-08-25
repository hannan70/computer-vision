from ultralytics import YOLO
import cv2
import math

# setup model
model = YOLO("yolo11n.pt")

# load video
# video = "./dataset/Videos/video8.mp4"

# tracking with default tracker with bot-sort
# results = model.track(source=video, show=True)

# tracking with default tracker with ByteTrack
# results = model.track(source=video, show=True, tracker="bytetrack.yaml")

# --------------- customize yolo

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


video = cv2.VideoCapture("./dataset/Videos/video4.mp4")

while True:
    ret, frame = video.read()

    if not ret: break

    results = model(frame, conf=0.30, iou=0.5, classes=[0])
    person_count = 0
    
    for res in results:
        boxes = res.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # draw a plot
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)

            # get classname
            classNameInt = int(box.cls[0])
            className = cocoClassNames[classNameInt]

            if className == "person":
                person_count += 1

            # calculate conf score
            conf = math.ceil(box.conf[0]*100)/100
            label = f"{className}: {conf}"

             # Background for label
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1, 2)
            cv2.rectangle(frame, (x1, y1 - h - 20), (x1+w, y1), (255, 0, 0), -1)

            # conf score
            cv2.putText(frame, label, (x1, y1-20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

           
        

    # Total person count on the frame
    cv2.putText(frame, f"Total Person:{person_count}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)


    cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video.release()
cv2.destroyAllWindows()

