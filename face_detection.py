import cv2
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
print("Press SPACE to save photo")
print("Press ESC to exit")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    results = model(frame, classes=[0])  # ✅ Filter: only detect 'person' (class 0)
    face_count = 0
    for result in results:
        for box in result.boxes:
            # ✅ Extra guard: skip if not class 0 (person)
            if int(box.cls[0]) != 0:
                continue
            confidence = float(box.conf[0])
            if confidence < 0.5:           # ✅ Skip low-confidence detections
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # ✅ Show confidence score above each box
            cv2.putText(
                frame,
                f"{confidence:.0%}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )
    cv2.putText(
        frame,
        f"Faces: {face_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()