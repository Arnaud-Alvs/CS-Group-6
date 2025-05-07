import pickle
import os

# Load the model, vectorizer, and label encoder
model_path = "waste_classifier.pkl"
vectorizer_path = "waste_vectorizer.pkl"
encoder_path = "waste_encoder.pkl"

# Make sure files exist
if not all([os.path.exists(model_path), os.path.exists(vectorizer_path), os.path.exists(encoder_path)]):
    raise FileNotFoundError("One or more model files are missing. Please train the model first.")

with open(model_path, "rb") as model_file:
    classifier = pickle.load(model_file)

with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open(encoder_path, "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Function to classify text
def classify_text(text):
    text_vectorized = vectorizer.transform([text])  # transform user text
    prediction_encoded = classifier.predict(text_vectorized)  # predict category
    prediction = label_encoder.inverse_transform(prediction_encoded)  # decode category
    return prediction[0]  # return the first result

# Example (for testing only, you can remove later)
if __name__ == "__main__":
    test_text = "aerosol"
    result = classify_text(test_text)
    print("Predicted category:", result)
