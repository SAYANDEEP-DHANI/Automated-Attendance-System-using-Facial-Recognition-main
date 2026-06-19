import re
import cv2
import numpy as np
import os
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from openpyxl import Workbook
from datetime import datetime
model = YOLO("yolov8n.pt")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))


known_embeddings = []
known_names = []
present_students = set()
known_folder = "FACES"
for file in os.listdir(known_folder):
    img_path = os.path.join(known_folder, file)
    img = cv2.imread(img_path)
    faces = app.get(img)
    if len(faces) > 0:
        embedding = faces[0].embedding
        known_embeddings.append(embedding)
        name = re.sub(r'\d+$', '', os.path.splitext(file)[0])
        known_names.append(name)
print("Known Faces Loaded:", known_names)


cap = cv2.VideoCapture(0)
def save_attendance():
    wb = Workbook()
    ws = wb.active

   # ws.append(["Name", "Date", "Time", "Status"])

    now = datetime.now()

    for student in present_students:
        ws.append([
            student,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            "Present"
        ])

    wb.save("attendance.xlsx")
    print("Attendance saved to attendance.xlsx")
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            detected_faces = app.get(face_crop)
            if len(detected_faces) > 0:
                embedding = detected_faces[0].embedding
                best_match = "Unknown"
                best_score = -1
                for known_emb, name in zip(
                        known_embeddings,
                        known_names):
                    similarity = np.dot(
                        embedding,
                        known_emb
                    ) / (
                        np.linalg.norm(embedding)
                        * np.linalg.norm(known_emb)
                    )
                    if similarity > best_score:
                        best_score = similarity
                        best_match = name
                if best_score < 0.30:
                    best_match = "Unknown"
                if best_match != "Unknown":
                    present_students.add(best_match)
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    frame,
                    f"{best_match} {best_score:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('a'):
        save_attendance()

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows()