from ultralytics import YOLO


def object_detection(image):
    # Load YOLOv8 model
    model = YOLO("best.pt")
    return model(image, conf=0.7, save_txt=True)


def check_detected(model_results):
    return bool(model_results[0].boxes)


def calculate_bbox_area(model_results):
    boxes = model_results[0].boxes
    area = boxes.xywh.tolist()[0][2] * boxes.xywh.tolist()[0][3]
    return area
