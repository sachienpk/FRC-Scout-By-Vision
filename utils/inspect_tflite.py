import tensorflow as tf

model_path = "B2_CPU_coral_and_algae_monochrome.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

print("== INPUT DETAILS ==")
for detail in interpreter.get_input_details():
    print(detail)

print("\n== OUTPUT DETAILS ==")
for detail in interpreter.get_output_details():
    print(detail)
