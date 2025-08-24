from ultralytics import YOLO

try:
    model = YOLO("yolov8n.pt")
    print("✅ YOLO model loaded successfully")
except Exception as e:
    print("❌ Error loading YOLO model:", e)
