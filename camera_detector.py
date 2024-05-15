import logging

import cv2

from services import object_detection, check_detected, calculate_bbox_area

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detection_flags = []
areas = []


def perform_detections():
    while True:
        ret, frame = cap.read()

        # Perform object detection
        logging.info("Performing object detection")
        results = object_detection(frame)

        # Update detection flag
        if flag := check_detected(results):
            logging.info("BOAT DETECTED")
            detection_flags.append(flag)
            areas.append(calculate_bbox_area(results))

        # plot results
        logging.info("Plotting results")
        frame_ = results[0].plot()

        # visualize
        cv2.imshow("Detecting", frame_)

        if cv2.waitKey(1) == ord('q'):
            break

        print(areas)

    cap.release()
    cv2.destroyAllWindows()


perform_detections()
logging.info(f"Detection completed. Found {len(detection_flags)} boats.")
