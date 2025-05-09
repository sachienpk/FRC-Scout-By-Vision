import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter

# === CONFIGURATION ===
MODEL_PATH = "/workspaces/FRC-Scout-By-Vision/models/B2_CPU_coral_and_algae_monochrome.tflite"
IMAGE_PATH = "/workspaces/FRC-Scout-By-Vision/models/Game-piece/coral.png"  # Replace with your test image
CONFIDENCE_THRESHOLD = 0.3       # Ignore very low scores

# === Load TFLite Model ===
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, input_height, input_width, _ = input_details[0]['shape']

# === Load and Prepare Image ===
image = cv2.imread(IMAGE_PATH)
resized = cv2.resize(image, (input_width, input_height))
input_data = np.expand_dims(resized, axis=0).astype(np.uint8)

# === Run Inference ===
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# === Get Output Tensors ===
classes = interpreter.get_tensor(output_details[0]['index'])[0]
boxes = interpreter.get_tensor(output_details[1]['index'])[0]
count = int(interpreter.get_tensor(output_details[2]['index'])[0])
scores = interpreter.get_tensor(output_details[3]['index'])[0]

# === Analyze Results ===
best_index = -1
best_score = 0
best_class = None

for i in range(count):
    if scores[i] > best_score and scores[i] >= CONFIDENCE_THRESHOLD:
        best_score = scores[i]
        best_class = int(classes[i])
        best_index = i

if best_class is not None:
    print(f"Most likely class ID: {best_class} (score: {best_score:.2f})")
else:
    print("No confident detections.")
