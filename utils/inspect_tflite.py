import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter

# ===== CONFIGURATION =====
MODEL_PATH = "/workspaces/FRC-Scout-By-Vision/models/B2_CPU_coral_and_algae_monochrome.tflite"
IMAGE_PATHS = [
    "/workspaces/FRC-Scout-By-Vision/models/Game-piece/coral.png",
    "/workspaces/FRC-Scout-By-Vision/models/Game-piece/algae.jpeg",
]
CONFIDENCE_THRESHOLD = 0.3

# ===== MODEL INSPECTION =====
def inspect_model(interpreter):
    print("\n===== MODEL INSPECTION =====")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\n[Input Details]")
    print(f"Name: {input_details[0]['name']}")
    print(f"Shape: {input_details[0]['shape']}")
    print(f"Type: {input_details[0]['dtype']}")
    
    print("\n[Output Details]")
    for i, detail in enumerate(output_details):
        print(f"Output {i}:")
        print(f"Name: {detail['name']}")
        print(f"Shape: {detail['shape']}")
        print(f"Type: {detail['dtype']}")

# ===== IMAGE ANALYSIS =====
def analyze_image(interpreter, image_path):
    print(f"\n===== ANALYZING: {image_path.split('/')[-1]} =====")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None, None
    
    # Prepare input
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    height, width = input_shape[1], input_shape[2]
    resized = cv2.resize(image, (width, height))
    input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Get outputs - based on your model inspection
    output_details = interpreter.get_output_details()
    
    # Output order from your model:
    # 0: StatefulPartitionedCall:1 (1,10) - likely classes
    # 1: StatefulPartitionedCall:3 (1,10,4) - likely boxes
    # 2: StatefulPartitionedCall:0 (1) - likely count
    # 3: StatefulPartitionedCall:2 (1,10) - likely scores
    
    classes = interpreter.get_tensor(output_details[0]['index'])[0]  # (10,)
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]    # (10,4)
    count = int(interpreter.get_tensor(output_details[2]['index'])[0])  # scalar
    scores = interpreter.get_tensor(output_details[3]['index'])[0]   # (10,)
    
    print("\n[Raw Outputs]")
    print(f"Number of detections: {count}")
    print(f"Classes: {classes[:count]}")
    print(f"Scores: {scores[:count]}")
    print(f"Boxes (first): {boxes[0]}")
    
    # Find best detection
    best_idx = -1
    best_score = 0
    best_class = None
    
    for i in range(count):
        if scores[i] > best_score and scores[i] >= CONFIDENCE_THRESHOLD:
            best_score = scores[i]
            best_class = int(classes[i])
            best_idx = i
    
    if best_idx != -1:
        print(f"\n[Best Detection]")
        print(f"Class: {best_class}")
        print(f"Score: {best_score:.4f}")
        print(f"Box: {boxes[best_idx]}")
    else:
        print("\nNo detections above confidence threshold")
    
    return best_class, best_score

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    try:
        # Initialize interpreter
        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        # Inspect model
        inspect_model(interpreter)
        
        # Process images
        results = {}
        for path in IMAGE_PATHS:
            class_id, score = analyze_image(interpreter, path)
            results[path.split('/')[-1]] = (class_id, score)
        
        # Print summary
        print("\n===== RESULTS SUMMARY =====")
        for img_name, (class_id, score) in results.items():
            if class_id is not None:
                print(f"{img_name}: Class {class_id} (Score: {score:.4f})")
            else:
                print(f"{img_name}: No detection")
        
        print("\n===== INTERPRETATION GUIDE =====")
        print("1. Class IDs:")
        print("   - 0: Coral (expected)")
        print("   - 1: Algae (expected)")
        print("2. If getting unexpected classes:")
        print("   - Check training labels matched these IDs")
        print("3. For low confidence scores:")
        print("   - Verify images match training data format")
        print("   - Check lighting/angles match training conditions")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nTROUBLESHOOTING:")
        print("1. Verify the model file exists at the specified path")
        print("2. Check all image paths are correct")
        print("3. Ensure TensorFlow Lite is properly installed")
        print("4. If using custom model, verify output format matches expected")