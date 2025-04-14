import cv2
import time
import numpy as np

# Load YOLOv5 (you'll need the pre-trained weights)
net = cv2.dnn.readNet("usecase_version_3\yolov5\yolov5s.pt", "data.yaml")

# Load COCO class names
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize variables
last_seen_water_time = time.time()
water_bottle_detected = False

while True:
    ret, frame = cap.read()

    # Detect objects using YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Check if water bottle is detected
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "bottle":
                water_bottle_detected = True
                last_seen_water_time = time.time()

    # Check dehydration status
    if not water_bottle_detected and time.time() - last_seen_water_time > 30:
        print("Uh-oh! You might be dehydrated. Drink some water!")

    # Display the frame
    cv2.imshow("Dehydration Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
