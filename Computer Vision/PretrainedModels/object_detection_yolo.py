import sys
import cv2
from ultralytics import YOLO

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODEL_NAME = "yolo11n.pt"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.5
WINDOW_NAME = "YOLO Webcam Object Detection"

model = YOLO(MODEL_NAME)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Webcam açılamadı. Kamera indeksini kontrol edin.")

print("Webcam başlatıldı. Çıkış için 'q' tuşuna basın.")

last_detected_labels = set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame okunamadı.")
        break

    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

    detections = []
    detected_labels = set()
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[class_id]
            detections.append(f"{label} ({confidence:.2f})")
            detected_labels.add(label)

    if detected_labels and detected_labels != last_detected_labels:
        print("Tespit edilen nesneler:", ", ".join(sorted(detections)))
        last_detected_labels = detected_labels

    annotated_frame = results[0].plot()
    cv2.imshow(WINDOW_NAME, annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
