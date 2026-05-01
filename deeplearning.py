import numpy as np
import cv2
import os
import pytesseract as pt

# =========================
# TESSERACT FIX (IMPORTANT)
# =========================
if os.name == "nt":
    pt.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pt.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# =========================
# MODEL CONFIG
# =========================
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

MODEL_PATH = "./static/models/best.onnx"

net = cv2.dnn.readNetFromONNX(MODEL_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# =========================
# DETECTION
# =========================
def get_detections(img, net):
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    blob = cv2.dnn.blobFromImage(
        input_image, 1/255,
        (INPUT_WIDTH, INPUT_HEIGHT),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)
    preds = net.forward()
    return input_image, preds[0]


# =========================
# NMS
# =========================
def non_maximum_supression(input_image, detections):
    boxes = []
    confidences = []

    h, w = input_image.shape[:2]

    for row in detections:
        conf = row[4]

        if conf > 0.4:
            cls = row[5]

            if cls > 0.25:
                cx, cy, bw, bh = row[:4]

                x = int((cx - bw/2) * (w / INPUT_WIDTH))
                y = int((cy - bh/2) * (h / INPUT_HEIGHT))
                bw = int(bw * (w / INPUT_WIDTH))
                bh = int(bh * (h / INPUT_HEIGHT))

                boxes.append([x, y, bw, bh])
                confidences.append(float(conf))

    if len(boxes) == 0:
        return [], [], []

    idx = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    if len(idx) == 0:
        return boxes, confidences, []

    return boxes, confidences, idx.flatten()


# =========================
# OCR (FIXED SAFE)
# =========================
def extract_text(image, bbox):
    x, y, w, h = bbox

    roi = image[y:y+h, x:x+w]

    if roi.size == 0:
        return ""

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    try:
        text = pt.image_to_string(gray, config="--psm 7")
        return text.strip()
    except:
        return ""


# =========================
# PIPELINE
# =========================
def yolo_predictions(img, net):
    inp, det = get_detections(img, net)
    boxes, conf, idx = non_maximum_supression(inp, det)

    texts = []
    final_boxes = []

    if len(idx) == 0:
        return texts, final_boxes

    for i in idx:
        bbox = boxes[i]
        txt = extract_text(img, bbox)

        texts.append(txt)
        final_boxes.append(bbox)

    return texts, final_boxes


# =========================
# ENTRY
# =========================
def object_detection(path, filename):
    img = cv2.imread(path)

    if img is None:
        return [], []

    return yolo_predictions(img, net)