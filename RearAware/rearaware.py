from ultralytics import YOLO
import cv2
import subprocess
import random
import time
import pyvirtualcam

# Load your model
model = YOLO("models/30-cfb.pt")

# Load the PNG (must have transparency)
censor = cv2.imread("censored.png", cv2.IMREAD_UNCHANGED)
if censor is None:
    raise FileNotFoundError("Couldn't find censored.png")

sounds = [
    "sounds/duck.wav",
    "sounds/fart1.wav",
    "sounds/fart2.wav",
    "sounds/fart3.wav",
    "sounds/fart4.wav",
    "sounds/fart5.wav",
    "sounds/fart6.wav",
    "sounds/fart7.wav",
    "sounds/fart8.wav",
    "sounds/fart9.wav",
    "sounds/fart10.wav",
    "sounds/fart11.wav",
    "sounds/fart12.wav",
    "sounds/law-and-order.wav",
    "sounds/mgs-alert.wav",
    "sounds/psycho.wav",
    "sounds/wasted.wav",
    "sounds/wilhelm.wav",
    "sounds/windows-error.wav",
]

last_sound_time = 0
sound_cooldown = 3.0
butt_visible_last_frame = False
frame_count = 0
last_results = None

# Open webcam
cap = cv2.VideoCapture(0)
time.sleep(1)

# Read one frame to get resolution
success, frame = cap.read()
if not success:
    raise RuntimeError("Couldn't read from webcam.")

height, width = frame.shape[:2]
print(f"Webcam: {width}x{height}")

cam = pyvirtualcam.Camera(width=1920, height=1080, fps=30)
print(f"Virtual camera: {cam.device}")
print("RearAware is running. Press Ctrl+C to stop.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    butt_detected_this_frame = False

    if frame_count % 2 == 0:
        small = cv2.resize(frame, (640, 360))
        last_results = model(small, verbose=False)

    if last_results is None:
        continue

    results = last_results

    for box in results[0].boxes:
        cls = int(box.cls[0])

        if cls == 2:
            butt_detected_this_frame = True
            current_time = time.time()

            if (
                not butt_visible_last_frame
                and current_time - last_sound_time > sound_cooldown
            ):
                sound = random.choice(sounds)
                subprocess.Popen(
                    ["afplay", sound],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                last_sound_time = current_time

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = int(x1 * frame.shape[1] / 640)
            x2 = int(x2 * frame.shape[1] / 640)
            y1 = int(y1 * frame.shape[0] / 360)
            y2 = int(y2 * frame.shape[0] / 360)
            box_w = x2 - x1
            box_h = y2 - y1

            img_h, img_w = censor.shape[:2]
            scale = (box_h * 2.0) / img_h
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)

            overlay = cv2.resize(censor, (new_w, new_h))

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            left = max(0, cx - new_w // 2)
            top = max(0, cy - new_h // 2)
            right = min(frame.shape[1], left + new_w)
            bottom = min(frame.shape[0], top + new_h)

            overlay = overlay[:bottom-top, :right-left]
            bgr = overlay[:, :, :3]
            alpha = overlay[:, :, 3] / 255.0
            roi = frame[top:bottom, left:right]

            for c in range(3):
                roi[:, :, c] = (
                    alpha * bgr[:, :, c] +
                    (1 - alpha) * roi[:, :, c]
                )
            frame[top:bottom, left:right] = roi

    butt_visible_last_frame = butt_detected_this_frame

    cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cam.sleep_until_next_frame()

cap.release()
