from ultralytics import YOLO


def object_detection(image):
    # Load YOLOv8 model
    model = YOLO("best.pt")
    return model(image)


def check_detected(model_results):
    return bool(model_results[0].boxes)
