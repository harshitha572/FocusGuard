# Focus Guard

Real-time system to detect people, cell phones, and earbuds/headphones using webcam.

---

## **Requirements**

- Python 3.10+  
- Packages: `opencv-python`, `ultralytics`, `mediapipe`, `numpy`

---

## **Setup**

```bash
git clone https://github.com/harshitha572/FocusGuard.git
cd FocusGuard
python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install opencv-python ultralytics mediapipe numpy
```

---

## **Run**

```bash
python focus_guard_full.py
```

- Green boxes → people  
- Red boxes → phones  
- Blue text → earbuds/headphones  
- Press **q** to exit
