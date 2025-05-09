import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter

class CoralAlgaeDetector:
    def __init__(self, model_path):
        # Initialize TFLite model
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()
        
        # Color thresholds (HSV ranges)
        self.CORAL_COLOR_RANGE = ((0, 100, 100), (15, 255, 255))  # Reddish hues
        self.ALGAE_COLOR_RANGE = ((40, 50, 50), (80, 255, 255))    # Greenish hues

    def detect(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Image load failed"}
        
        # Run object detection
        boxes, classes, confidences = self._run_detection(img)
        
        # Analyze color when detection is ambiguous
        results = []
        for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
            if conf < 0.7:  # Low confidence - use color verification
                color_class = self._color_analysis(img, box)
                final_class = color_class if color_class else cls
            else:
                final_class = cls
            
            results.append({
                "box": box,
                "class": final_class,
                "confidence": conf,
                "method": "color" if conf < 0.7 else "model"
            })
        
        return results

    def _run_detection(self, img):
        """Runs standard object detection"""
        # Preprocess
        input_shape = self.input_details['shape'][1:3]
        resized = cv2.resize(img, (input_shape[1], input_shape[0]))
        input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
        
        # Inference
        self.interpreter.set_tensor(self.input_details['index'], input_data)
        self.interpreter.invoke()
        
        # Get outputs (adjust indices as needed)
        boxes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        confidences = self.interpreter.get_tensor(self.output_details[3]['index'])[0]
        
        return boxes, classes, confidences

    def _color_analysis(self, img, box):
        """Determines class based on dominant color in bounding box"""
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Extract ROI
        y1, x1, y2, x2 = box
        roi = hsv[int(y1*img.shape[0]):int(y2*img.shape[0]), 
                   int(x1*img.shape[1]):int(x2*img.shape[1])]
        
        # Calculate color histograms
        coral_mask = cv2.inRange(roi, *self.CORAL_COLOR_RANGE)
        algae_mask = cv2.inRange(roi, *self.ALGAE_COLOR_RANGE)
        
        coral_pixels = np.count_nonzero(coral_mask)
        algae_pixels = np.count_nonzero(algae_mask)
        
        if coral_pixels > algae_pixels * 1.5:  # Significant coral color
            return 0
        elif algae_pixels > coral_pixels * 1.5:  # Significant algae color
            return 1
        return None  # Inconclusive

# Usage
detector = CoralAlgaeDetector("/workspaces/FRC-Scout-By-Vision/models/B2_CPU_coral_and_algae_monochrome.tflite")
print("Coral Result:", detector.detect("/workspaces/FRC-Scout-By-Vision/models/Game-piece/coral.png"))
print("Algae Result:", detector.detect("/workspaces/FRC-Scout-By-Vision/models/Game-piece/algae.jpeg"))
