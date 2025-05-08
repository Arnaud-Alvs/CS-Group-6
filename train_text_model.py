# train_text_model.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Step 1: Load the expanded dataset
df = pd.read_csv("waste_text_dataset_expanded.csv")

# Step 2: Prepare features (X) and labels (y)
X = df["Description"]  # text input
y = df["Category"]     # target class

# Step 3: Encode the target labels (so the model works with numbers)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 4: Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Step 5: Train the classification model
classifier = RandomForestClassifier(n_estimators=200, random_state=42)
classifier.fit(X_vectorized, y_encoded)

# Step 6: Save the trained model, vectorizer and label encoder for later use in app.py
with open("waste_classifier.pkl", "wb") as model_file:
    pickle.dump(classifier, model_file)

with open("waste_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

with open("waste_encoder.pkl", "wb") as encoder_file:
    pickle.dump(label_encoder, encoder_file)

print("✅ Text classification model, vectorizer, and encoder saved successfully!")