from os.path import exists
from detection.object_detection import detect_object
from PIL import Image
import numpy as np


def test_yolov_cfg_exist():
    assert exists("yolo/yolov4.cfg")

def test_yolov_coco_exist():
    assert exists("yolo/coco.names")

def test_model_work_jpg():
    image = Image.open("test_image.jpg")
    image = image.convert("RGB")
    img_array = np.array(image)
    detection = detect_object(img_array)
    classes = detection[0]
    confidences = detection[1]
    print(classes, confidences)
    assert classes[0] == 0
    assert classes[1] == 0
    assert classes[2] == 0
    assert classes[3] == 0
    assert isclose(confidences[0], 0.9932654)
    assert isclose(confidences[1], 0.98865324)
    assert isclose(confidences[2], 0.9708028)
    assert isclose(confidences[3], 0.9642243)

def test_model_work_png():
    image = Image.open("test_image.png")
    image = image.convert("RGB")
    img_array = np.array(image)
    detection = detect_object(img_array)
    classes = detection[0]
    confidences = detection[1]
    print(classes, confidences)
    assert classes[0] == 0
    assert classes[1] == 0
    assert isclose(confidences[0], 0.91026515)
    assert isclose(confidences[1], 0.3054578)

def isclose(a, b, rel_tol=1e-05, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)