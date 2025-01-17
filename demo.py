import tensorflow as tf
from tensorflow.keras.models import load_model

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

# Test loading the model
try:
    model = load_model("Model/keras_model.h5", compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
