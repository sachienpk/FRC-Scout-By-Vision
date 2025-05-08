import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter

# === CONFIGURATION ===
VIDEO_PATH = "/workspaces/FRC-Scout-By-Vision/match-video/video.mp4"                 # Replace with your video file path
MODEL_PATH = "/workspaces/FRC-Scout-By-Vision/models/B2_CPU_coral_and_algae_monochrome.tflite"
DETECTION_THRESHOLD = 0.5
LABELS = ["coral", "algae"]              # Adjust if needed
TARGET_CLASS_ID = None                   # Set to 0 or 1 to count only coral or algae

# === Load TFLite Model ===
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, input_height, input_width, _ = input_details[0]['shape']

# === Process Video ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
    exit(1)

frame_index = 0
total_count = 0

print("[INFO] Starting inference...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and preprocess frame
    resized = cv2.resize(frame, (input_width, input_height))
    input_data = np.expand_dims(resized, axis=0).astype(np.uint8)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Extract outputs
    classes = interpreter.get_tensor(output_details[0]['index'])[0]
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    count = interpreter.get_tensor(output_details[2]['index'])[0]
    scores = interpreter.get_tensor(output_details[3]['index'])[0]

    detected = 0
    for i in range(int(count)):
        if scores[i] >= DETECTION_THRESHOLD:
            if TARGET_CLASS_ID is None or int(classes[i]) == TARGET_CLASS_ID:
                detected += 1

    total_count += detected
    print(f"[FRAME {frame_index}] Detected {detected} object(s)")
    frame_index += 1

cap.release()
print(f"[RESULT] Total objects detected in video: {total_count}")
