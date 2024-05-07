import logging

import cv2
from ultralytics import YOLO

ESC_KEY = 27
VIDEO = 3

# Load YOLOv8 model
model = YOLO("best.pt")

videos = {1: {"path": "prodromos_2021_10_29_sailboats_busy/videos/2",
              "perspective": "right"},
          2: {"path": "philos_2021_10_28_dusk_mooring_field/videos",
              "perspective": "right"},
          3: {"path": "philos_2021_10_28_dusk_party_boat_glare/videos",
              "perspective": "center"}
          }

video = cv2.VideoCapture(
    f"mit-marine-perception-dataset/"
    f"{videos[VIDEO].get('path')}/{videos[VIDEO].get('perspective')}_camera_VID.mp4")

while True:
    logging.info("Reading video")
    success, frame = video.read()
    if not success:
        logging.error("Failed while reading video")
        break

    # Perform object detection
    logging.info("Performing object detection")
    results = model(frame)

    # plot results
    logging.info("Plotting results")
    frame_ = results[0].plot()

    # visualize
    cv2.imshow("Detecting", frame_)

    if cv2.waitKey(1) == ESC_KEY:
        break

# TODO: salvar a area fora do loop pra fazer a diferença