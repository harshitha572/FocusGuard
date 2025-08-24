import cv2
from ultralytics import YOLO

# --- Webcam setup (working combination) ---
camera_index = 0
backend = cv2.CAP_DSHOW

cap = cv2.VideoCapture(camera_index, backend)
if not cap.isOpened():
    print(f"❌ Cannot open webcam with index {camera_index} and backend {backend}")
    exit()

# --- Load YOLOv8 model ---
try:
    model = YOLO("yolov8n.pt")  # downloads automatically if needed
    print("✅ YOLO model loaded")
except Exception as e:
    print("❌ Error loading YOLO model:", e)
    exit()

# --- Helper function ---
def near_head(phone_box, person_box):
    """Return True if phone is near the top 20% of person bbox (head)."""
    x1, y1, x2, y2 = person_box
    head_limit = y1 + 0.2 * (y2 - y1)
    phone_center_y = (phone_box[1] + phone_box[3]) / 2
    return phone_center_y < head_limit

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLO detection
    results = model(frame, conf=0.5)

    persons, phones = [], []
    for r in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = r
        cls = int(cls)
        if model.names[cls] == "person":
            persons.append((int(x1), int(y1), int(x2), int(y2)))
        elif model.names[cls] == "cell phone":
            phones.append((int(x1), int(y1), int(x2), int(y2)))

    # Draw boxes and alerts
    for p in persons:
        cv2.rectangle(frame, (p[0], p[1]), (p[2], p[3]), (0,255,0), 2)
        for ph in phones:
            cv2.rectangle(frame, (ph[0], ph[1]), (ph[2], ph[3]), (0,0,255), 2)
            if near_head(ph, p):
                cv2.putText(frame, "⚠️ Phone near face!", (p[0], p[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            else:
                cv2.putText(frame, "Phone in hand", (p[0], p[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.imshow("FocusGuard", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
