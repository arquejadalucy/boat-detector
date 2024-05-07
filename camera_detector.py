import logging

import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


def object_detection(image):
    # Load YOLOv8 model
    model = YOLO("best.pt")
    return model(image)


def check_detected(model_results):
    return bool(model_results[0].boxes)


detection_flags = []

while True:
    ret, frame = cap.read()

    # Perform object detection
    logging.info("Performing object detection")
    results = object_detection(frame)

    # Update detection flag
    if flag := check_detected(results):
        logging.info("BOAT DETECTED")
        detection_flags.append(flag)

    # plot results
    logging.info("Plotting results")
    frame_ = results[0].plot()

    # visualize
    cv2.imshow("Detecting", frame_)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

logging.info(f"Detection completed. Found {len(detection_flags)} boats.")
