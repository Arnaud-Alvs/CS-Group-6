# This is a machine learning model that allows to recognise the waste type based on a text description,
# This is possible because we transform text into numbers, making it understandable by the computer.
# We have trained it thanks to a model and use it in the main app.py

# We import the necessary libraries in order to run properly the code
# We import pandas, which as saw in the lecture 8 of the course as the most popular library for dealing and manipulating csv files (in this case our dataset waste_text_dataset_expanded.csv)
#Further we used TfidVecotrizer and Labelencoder to respectively convert the text descriptions into numerical format, and to transform out waste categories into numbers.
#pickle was used instead, as in other files of our project, in order to be able to later upload successfully the trained model into app.py
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# We load the csv file (our dataset) that contains a data set of descriptions matching with waste categories (example : pizza box -> matches with the cardboard category).
# We have created then our dataframe "df" by loading the column and rows of waste_text_dataset_expanded.csv.
df = pd.read_csv("waste_text_dataset_expanded.csv")

# We separate the the data : 
# -x is used for the description part (example : pizza box)
# -y is used for the category part (exemple : cadrboard)
X = df["Description"]  # text input
y = df["Category"]     # target class

# We transform the categories into numbers (since the model doesn't understand text), by doing this, we are associating a number to each category
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# We Convert text into numerical features using TF-IDF, since as before, the model is not able to recognize the text.
# For developing this part of the code we directly asked generative AI to provide a suitable library TF-IDF gives higher weight to important words
# In this case we have followed the suggestion of ChatGPT to use 5000 words to keep the model lightweight.
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# We create a Random Forest model: a ready to use algorith for machine learning wich combines the reuslt of multiple
# decision trees in order to provide us with a single, most suitable, result to classify the data.
classifier = RandomForestClassifier(n_estimators=200, random_state=42) # We have used 200 estimators for ensuring sufficient stability and so called fixed seed to make the code reproducible
classifier.fit(X_vectorized, y_encoded) # We train the model with the texts (which have been turned into numbers) and check if it matches the 
# corresponding categories

# We save the trained model in order to benefit from a trained version we needed. IMPORTANT: don't forget that the model has to be re-trained everytime 
# we changed our dataset file waste_text_dataset_expaneded.csv, since it has to train again when we have changed/add/removev items from the list
with open("waste_classifier.pkl", "wb") as model_file:
    pickle.dump(classifier, model_file)

# We save the model that transforms text into numbers
with open("waste_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
with open("waste_encoder.pkl", "wb") as encoder_file:#
    pickle.dump(label_encoder, encoder_file)

print("pkl files from the train_text_model have been saved succesfully") #just to be sure that the model has been trained accordingly to what coded above

# With support from ChatGPT (OpenAI), consulted for debugging and resolving initial implementation errors - Andreas Lucchini.
