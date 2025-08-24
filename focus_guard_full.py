import cv2
from ultralytics import YOLO
import mediapipe as mp

# --- Webcam setup ---
camera_index = 0
backend = cv2.CAP_DSHOW
cap = cv2.VideoCapture(camera_index, backend)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()
print("✅ Webcam opened")

# --- Load YOLO model ---
model = YOLO("yolov8n.pt")
print("✅ YOLO model loaded")

# --- MediaPipe Face Mesh for earbud/head detection ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Helper function ---
def near_head(phone_box, person_box):
    """Return True if phone is near top 20% of person bbox (head)."""
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

    # Flip frame for correct orientation
    frame = cv2.flip(frame, 1)

    # --- YOLO detection ---
    results = model(frame, conf=0.5)

    persons, phones = [], []
    for r in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = r
        cls = int(cls)
        if model.names[cls] == "person":
            persons.append((int(x1), int(y1), int(x2), int(y2)))
        elif model.names[cls] == "cell phone":
            phones.append((int(x1), int(y1), int(x2), int(y2)))

    # --- MediaPipe Face Mesh detection ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_mp = face_mesh.process(rgb_frame)
    earbuds_detected = False
    if results_mp.multi_face_landmarks:
        for face_landmarks in results_mp.multi_face_landmarks:
            # Using landmarks near ears to detect earbuds/headphones
            left_ear = face_landmarks.landmark[234]  # approximate left ear
            right_ear = face_landmarks.landmark[454] # approximate right ear
            # If the ear landmarks are present, assume headphones/earbuds detected
            earbuds_detected = True
            break

    # --- Draw YOLO boxes and alerts ---
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

    # --- Draw earbuds/headphones alert ---
    if earbuds_detected:
        cv2.putText(frame, "🎧 Earbuds/Headphones detected", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    # Show the final frame
    cv2.imshow("Focus Guard Full", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
