# 📦 Streamlit App Groupe des BG 
```
⬆️ (Nom de l'app)
```

Description de l'app:

Solution: Apportez aux utilisateurs une interface leur permettant de détérminer le trajet optimal pour aller jetter leur poubelle.

L'idée serait que les utilisateurs puissent donner leur poubelle, et le programme détérminerait le trajet optimal. 

Les API, se trouvent dans ce lien:  https://daten.stadt.sg.ch/explore/dataset/abfuhrdaten-stadt-stgallen/information/?disjunctive.gebietsbezeichnung&disjunctive.sammlung&disjunctive.strasse




## Demo App

Lien de l'app sur streamlit (a voir pour plus tard)

## GitHub Codespaces

Pour les codespaces (et plus particulièrement le travail en groupe) comment ça marche ? 

Premièrement, chacun travaillera sur une branche particulière. 
Deuxièmement, les codespaces ne se partagent pas et "work" locally, ce qui veut dire que chaque changement fais sur votre codespace doit être push a la branche correspondante et qu'à chaque fois que vous y retravaillez dessus il faut le mettre a jour (dans le cas ou des changements ont été éffctué par d'autre personne. 

Code pour commit:  
---  
git add .  
git commit -m "Implemented feature X"  
git push origin feature-name  

Code pour remettre le code a niveau:  
---  
git checkout main  
git pull origin main  
git checkout feature-name    
git merge main  

Et voila, pour l'instant c'est tout ce que je sais.
## Infos Importantes
Ne pas toucher a la branch "main". 

Proposition par le pitt : 

# ♻️ Smart Waste Classification App

This project is a Streamlit web application that helps users identify the appropriate waste category for an item, based on either a **text description** or an **image**. Additionally, the app offers tools like a **BAC (Blood Alcohol Content) calculator** and a **collection point locator** based on user location.

---

## 🚀 Features

- 📝 **Text-based Waste Classification**: Users can describe an item in text (e.g., "pizza box") and receive its waste category.
- 📷 **Image-based Recognition**: Upload a photo of the waste, and the app suggests its type.
- 🧭 **Collection Point Finder**: Locate nearby recycling/disposal locations for a specific waste type.
- 🚗 **Alcohol Tester Tool**: Calculates BAC and indicates whether the user is safe to drive.
- 🗺️ **Interactive Map**: View and interact with waste collection points on a map.

---

## 🧠 How It Works

### Machine Learning
- A **Random Forest Classifier** is used for text classification.
- The text is vectorized using **TF-IDF** (term frequency-inverse document frequency).
- A label encoder translates categories into numerical labels.
- All models are trained using a dataset: `waste_text_dataset_expanded.csv`

### APIs
- The app uses external APIs to retrieve location-based data (collection points, dates, etc.)

---

## 🧰 Technologies Used

| Tool | Purpose |
|------|---------|
| **Python** | Core programming language |
| **Streamlit** | Web interface |
| **Scikit-learn** | ML model training (Random Forest, LabelEncoder, TF-IDF) |
| **TensorFlow** | (Optional) For image classification |
| **Pandas / NumPy** | Data manipulation |
| **Folium** | Interactive maps |
| **Requests / JSON** | API communication |
| **Pickle** | Saving/loading ML models |

---

## 📁 Project Structure

```
├── app.py                      # Main application logic
├── train_text_model.py        # Script to train the text ML model
├── location_api.py            # Functions for location-based queries and map generation
├── text_classifier.py         # Classifies text inputs
├── .streamlit/                # Streamlit configuration
├── pages/
│   ├── 1_Home.py              # Home page
│   ├── 2_Find_Collection_Points.py
│   ├── 3_Identify_Waste.py
│   ├── 4_Alcoholtester.py
├── waste_classifier.pkl       # Saved ML model
├── waste_vectorizer.pkl       # TF-IDF vectorizer
├── waste_encoder.pkl          # Label encoder
├── waste_text_dataset_expanded.csv
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Set up a Virtual Environment (recommended)
```bash
python -m venv env
source env/bin/activate  # or .\env\Scripts\activate on Windows
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

### 4. Launch the App
```bash
streamlit run app.py
```

---

## 🤝 Credits

Developed by:
- Alves Arnaud
- Noah
- Andreas Lucchini

With support from:
- Course material (Lecture 8, etc.)
- ChatGPT (OpenAI) for code debugging and ideas

---

## 📜 License

This project is for educational purposes only and not intended for commercial use.

