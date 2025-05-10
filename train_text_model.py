# This is a machine learning model that allows to recognise the waste type based on a text description
# This is possible because we transform text into numbers, making it understandable by the computer.
# We have trained it thanks to a model and use it in the main app.py

# We import the necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# We load the csv file that contains a data set of descriptions matching with waste categories (example : pizza box -> matches with cardboard)
df = pd.read_csv("waste_text_dataset_expanded.csv")

# We separate the the data : 
# -x is used for the description part (example : pizza box)
# -y is used for the category part (exemple : cadrboard)
X = df["Description"]  # text input
y = df["Category"]     # target class

# Step 3: We transform the categories into numbers (since the model doesn't understand text)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# We create a Random Forest model to classify the data
classifier = RandomForestClassifier(n_estimators=200, random_state=42)
classifier.fit(X_vectorized, y_encoded) # We train the model with the texts (which have been turned into numbers) and check if it matches the corresponding categories

# We save the trained model
with open("waste_classifier.pkl", "wb") as model_file:
    pickle.dump(classifier, model_file)

# We save the model that transforms text into numbers
with open("waste_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
with open("waste_encoder.pkl", "wb") as encoder_file:
    pickle.dump(label_encoder, encoder_file)

# Display this message to confirm that everything was saved successfully
print("✅ Text classification model, vectorizer, and encoder saved successfully!")