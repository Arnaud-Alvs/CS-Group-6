from tensorflow.keras.models import load_model

try:
    model = load_model("waste_image_classifier.h5")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
