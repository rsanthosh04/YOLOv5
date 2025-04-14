import torch
import cv2
import numpy as np
import time

model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs\train\exp2\weights\best.pt')  
cap = cv2.VideoCapture(0)  
confidence_threshold = 0.4  
start_time = None
elapsed_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()
    rajesh_detected = False
    for *xyxy, conf, cls in detections:
        if conf >= confidence_threshold:  
            label = f'{model.names[int(cls)]} {conf:.2f}'
            if model.names[int(cls)] == 'Rajesh':
                rajesh_detected = True
            xyxy = list(map(int, xyxy))
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if rajesh_detected:
        if start_time is not None:
            elapsed_time += time.time() - start_time
            start_time = None
    else:
        if start_time is None:
            start_time = time.time()

    total_elapsed_time = elapsed_time
    if start_time is not None:
        total_elapsed_time += time.time() - start_time

    cv2.putText(frame, f'Time: {total_elapsed_time:.2f}s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('YOLOv5 Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
