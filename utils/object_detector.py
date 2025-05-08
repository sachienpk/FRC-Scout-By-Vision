import cv2
import numpy as np
from flask import Flask, Response, request, render_template_string
from threading import Thread, Lock
from tensorflow.lite.python.interpreter import Interpreter
import time

# === CONFIGURATION ===
MODEL_PATH = "/workspaces/FRC-Scout-By-Vision/models/B2_CPU_coral_and_algae_monochrome.tflite"
VIDEO_SOURCES = {
    "Match Video": "/workspaces/FRC-Scout-By-Vision/match-video/video.mp4",
    "Webcam": 0
}
CLASS_NAMES = {1: "algae", 0: "coral"}  # Edit if class IDs differ
LABEL_COLORS = {
    "algae": (0, 255, 0),
    "coral": (255, 0, 0)
}
DETECTION_THRESHOLD = 0.3

# === Load TFLite Model ===
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, input_height, input_width, _ = input_details[0]['shape']

# === App Setup ===
app = Flask(__name__)
output_frame = None
frame_lock = Lock()
pause_detection = False
selected_class_id = 1
current_source = list(VIDEO_SOURCES.values())[0]
cap = cv2.VideoCapture(current_source)

# === Detection Thread ===
def detect():
    global output_frame, cap, pause_detection, selected_class_id, current_source

    prev_time = time.time()
    while True:
        if pause_detection:
            time.sleep(0.05)
            continue

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        resized = cv2.resize(frame, (input_width, input_height))
        input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Correct output ordering
        scores = interpreter.get_tensor(output_details[0]['index'])[0]
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]
        count = interpreter.get_tensor(output_details[2]['index'])[0]
        classes = interpreter.get_tensor(output_details[3]['index'])[0]

        height, width, _ = frame.shape

        for i in range(int(count)):
            if scores[i] >= DETECTION_THRESHOLD and int(classes[i]) == selected_class_id:
                y_min, x_min, y_max, x_max = boxes[i]
                x_min = int(x_min * width)
                x_max = int(x_max * width)
                y_min = int(y_min * height)
                y_max = int(y_max * height)
                label = CLASS_NAMES.get(selected_class_id, "object")
                color = LABEL_COLORS.get(label, (0, 255, 255))

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # FPS counter
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        with frame_lock:
            output_frame = frame.copy()

# === Stream Generator ===
def generate_stream():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n')

# === Flask Routes ===
@app.route('/')
def index():
    source_options = ''.join(
        f'<option value="{path}" {"selected" if path == current_source else ""}>{name}</option>'
        for name, path in VIDEO_SOURCES.items())
    class_options = ''.join(
        f'<option value="{cid}" {"selected" if cid == selected_class_id else ""}>{name}</option>'
        for cid, name in CLASS_NAMES.items())
    return render_template_string('''
        <h1>Algae/Coral Detector</h1>
        <form method="POST" action="/control">
            <label>Class to detect:</label>
            <select name="class_id">{{ class_options|safe }}</select><br><br>
            <label>Video source:</label>
            <select name="video_source">{{ source_options|safe }}</select><br><br>
            <button type="submit" name="action" value="update">Update</button>
            <button type="submit" name="action" value="pause">{{ 'Resume' if paused else 'Pause' }}</button>
        </form>
        <hr>
        <img src="/video" width="720">
    ''', source_options=source_options, class_options=class_options, paused=pause_detection)

@app.route('/video')
def video():
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST'])
def control():
    global pause_detection, selected_class_id, cap, current_source

    action = request.form.get("action")
    if action == "pause":
        pause_detection = not pause_detection
    elif action == "update":
        selected_class_id = int(request.form.get("class_id", selected_class_id))
        new_source = request.form.get("video_source", current_source)
        if new_source != current_source:
            cap.release()
            cap = cv2.VideoCapture(new_source)
            current_source = new_source
    return index()

# === Run App ===
if __name__ == "__main__":
    Thread(target=detect, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True)
 