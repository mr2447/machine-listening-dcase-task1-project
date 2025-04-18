import os
import numpy as np
from glob import glob
import tensorflow as tf

FEATURE_DIR = 'features/audio.2'
MODEL_PATH = 'Quantized_model/C1/converted_quant_model_default.tflite'

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model expects input shape:", input_details[0]['shape'])
feature_files = sorted(glob(os.path.join(FEATURE_DIR, '*.npy')))
print(f"✅ Loaded model: {MODEL_PATH}")
print(f"✅ Found {len(feature_files)} feature files")

for f in feature_files:
    x = np.load(f, allow_pickle=True)

    # Check if data has correct size
    if x.size != 40 * 51:
        print(f"❌ Skipping malformed file: {os.path.basename(f)}, shape: {x.shape}, size: {x.size}")
        continue

    # Ensure dtype and shape
    x = np.array(x, dtype=np.float32)
    x = np.reshape(x, (1, 40, 51, 1))  # Model expects this shape

    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = np.argmax(output_data)
    print(f"{os.path.basename(f)} → predicted class: {predicted_class}")