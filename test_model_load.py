# This file acts like a quick health check to make sure your saved AI model for image classification is ready to be used in the main app.
# It helps to verify that the saved model file is still valid (the .pkl files)

# This function loads our previously saved AI model classifiying waste based on images.
from tensorflow.keras.models import load_model

# This line tries to load the model from the file named "waste_image_classifier.h5".
# This file should contain a trained image recognition model.
try:
    model = load_model("waste_image_classifier.h5")
     # If no errors happen, we print a success message to say the model was loaded properly.
    print("✅ Model loaded successfully")
    # If something goes wrong (for example: the file doesn’t exist or is corrupted), this line catches the error and prints a message showing what went wrong.
except Exception as e:
    print(f"❌ Failed to load model: {e}")
