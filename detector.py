import logging

import cv2

from services import object_detection, check_detected, calculate_bbox_area

ESC_KEY = 27
VIDEO = 3

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

detection_flags = []
areas = []

while True:
    logging.info("Reading video")
    success, frame = video.read()
    if not success:
        logging.error("Failed while reading video")
        break

    # Perform object detection
    logging.info("Performing object detection")
    results = object_detection(frame)

    # Update detection flag and areas
    if flag := check_detected(results):
        logging.info("BOAT DETECTED")
        detection_flags.append(flag)
        bbox_area = calculate_bbox_area(results)
        areas.append(calculate_bbox_area(results))

    # plot results
    logging.info("Plotting results")
    frame_ = results[0].plot()

    # visualize
    cv2.imshow("Detecting", frame_)

    if cv2.waitKey(1) == ESC_KEY:
        break

logging.info(f"Detection completed. Found {len(detection_flags)} boats.")

# TODO: salvar a area fora do loop pra fazer a diferença