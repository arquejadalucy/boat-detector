import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from ultralytics import YOLO

ESC_KEY = 27
VIDEO = 2

class_name = 'Boat'

# Load YOLOv8 model
model = YOLO("best.pt")

videos = {1: {"path": "prodromos_2021_10_29_sailboats_busy/videos/2",
              "perspective": "right"},
          2: {"path": "philos_2021_10_28_dusk_mooring_field/videos",
              "perspective": "right"},
          3: {"path": "philos_2021_10_28_dusk_party_boat_glare/videos",
              "perspective": "center"}
          }

right_camera_video = cv2.VideoCapture(
    f"mit-marine-perception-dataset/"
    f"{videos[VIDEO].get('path')}/{videos[VIDEO].get('perspective')}_camera_VID.mp4")


while True:
    success, frame = right_camera_video.read()
    if not success:
        break

    # Perform object detection
    results = model(frame)
    results = results[0]
    bboxes = np.array(results.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(results.boxes.cls.cpu(), dtype="int")

    # display bboxes and labels
    for bbox in bboxes:
        (x1, y1, x2, y2) = bbox
        area = (x2-x1)*(y2-y1)

        # Extract the region from the frame
        roi = frame[y1:y2, x1:x2]
        roi = cv2.resize(roi, (225, 180))
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 226), 2)
        cv2.putText(frame, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)
    if key == ESC_KEY:
        break
