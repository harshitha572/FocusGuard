import cv2

camera_index = 0
backend = cv2.CAP_DSHOW

cap = cv2.VideoCapture(camera_index, backend)
if not cap.isOpened():
    print(f"❌ Cannot open webcam with index {camera_index}")
    exit()

print("✅ Webcam opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
