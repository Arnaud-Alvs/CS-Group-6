# This script allows the application to use a machine learning model that was already trained to classify waste items based on a text description (e.g., “pizza box” → “cardboard”)
# It loads three files (the trained model that can make predictions, a vectorizer that transforms text into numbers and a label encoder that decodes the model's output into a human-readable category)
# The function classify_text is created so that we can reuse this model without having to use all the text again

# Import the necessary modules
import pickle # Used to load files that store Python objects like models
import os # Used to check if files exist in the current folder

# These are the files generated when the model was trained and saved
model_path = "waste_classifier.pkl" # The trained classification model
vectorizer_path = "waste_vectorizer.pkl" # The tool to convert text into numbers
encoder_path = "waste_encoder.pkl" # The tool to convert numbers back into category names

# Check if all necessary files exist before continuing
# This avoids errors if someone tries to run the app without training the model first
if not all([os.path.exists(model_path), os.path.exists(vectorizer_path), os.path.exists(encoder_path)]):
    raise FileNotFoundError("One or more model files are missing. Please train the model first.")

# Load the trained classifier model from file
with open(model_path, "rb") as model_file:
    classifier = pickle.load(model_file)

# Load the TF-IDF vectorizer (it turns text into a format the model understands)
with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load the label encoder this translates model outputs back into category names
with open(encoder_path, "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Function to classify a text input
# Takes a user input like "glass bottle" and returns something like "Glass"
def classify_text(text):
    # Step 1: Convert the input text into a numerical vector using the same vectorizer used during training
    text_vectorized = vectorizer.transform([text])
    # Step 2: Use the trained model to predict the encoded category
    prediction_encoded = classifier.predict(text_vectorized)
    # Step 3: Convert the encoded number back into a readable label (like "Glass", "Plastic", etc.)
    prediction = label_encoder.inverse_transform(prediction_encoded)  # decode category
    # Step 4: Return the first result (since we only input one text string)
    return prediction[0]  # return the first result

# Example : useful for debugging or checking that everything works
if __name__ == "__main__":
    test_text = "aerosol"
    result = classify_text(test_text)
    print("Predicted category:", result)
