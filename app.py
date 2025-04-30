import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests
import json
import pickle
from sklearn.ensemble import RandomForestClassifier

# Configuration de la page
st.set_page_config(
    page_title="TriDéchets - Votre assistant de recyclage",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre de l'application
st.title("TriDéchets - Votre assistant de recyclage intelligent")
st.markdown("### Trouvez facilement où jeter vos déchets et contribuez à un environnement plus propre")

# Fonction pour charger le modèle ML (nous l'implémenterons plus tard)
@st.cache_resource
def load_model():
    try:
        with open('waste_classifier.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.warning("Le modèle n'est pas encore disponible. Certaines fonctionnalités seront limitées.")
        return None

# Interface principale avec onglets
tab1, tab2, tab3 = st.tabs(["Localiser un point de collecte", "Identifier un déchet", "À propos"])

with tab1:
    st.header("Trouvez où jeter vos déchets")
    
    # Demander le type de déchet
    waste_type = st.selectbox(
        "Quel type de déchet souhaitez-vous jeter ?",
        ["Papier/Carton", "Verre", "Plastique", "Métal", "Électronique", "Textile", "Dangereux", "Organique", "Autre"]
    )
    
    # Demander la localisation
    user_location = st.text_input("Entrez votre adresse ou code postal")
    
    if st.button("Rechercher les points de collecte"):
        if user_location:
            st.info("Nous recherchons les points de collecte proches de votre localisation...")
            # Ici nous ferons l'appel à l'API pour trouver les points de collecte
            # Pour l'instant, nous affichons un message fictif
            st.success(f"Plusieurs points de collecte pour {waste_type} ont été trouvés près de {user_location}")
            
            # Affichage d'une carte fictive pour le moment
            st.map(pd.DataFrame({
                'lat': [48.8566],
                'lon': [2.3522]
            }))
        else:
            st.error("Veuillez entrer une adresse ou un code postal")

with tab2:
    st.header("Identifier votre déchet")
    
    # Option pour décrire le déchet
    waste_description = st.text_area("Décrivez votre déchet (matériau, taille, utilisation, etc.)")
    
    # Option pour télécharger une image
    uploaded_file = st.file_uploader("Ou téléchargez une photo de votre déchet", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image téléchargée", width=300)
    
    if st.button("Identifier"):
        # Ici nous utiliserons le modèle ML pour identifier le type de déchet
        # Pour l'instant, nous affichons une réponse fictive
        if waste_description or uploaded_file:
            st.success("D'après notre analyse, il s'agit probablement d'un déchet recyclable de type PLASTIQUE")
            st.info("Conseils de tri : Ce déchet doit être jeté dans la poubelle jaune pour le recyclage")
        else:
            st.error("Veuillez décrire votre déchet ou télécharger une image")

with tab3:
    st.header("À propos de TriDéchets")
    st.write("""
    TriDéchets est une application éducative développée dans le cadre d'un projet universitaire.
    
    Notre mission est de simplifier le tri des déchets et de contribuer à un meilleur recyclage.
    
    L'application utilise des techniques d'intelligence artificielle pour identifier les déchets
    et vous guider vers les points de collecte les plus proches.
    """)

# Pied de page
st.markdown("---")
st.markdown("© 2025 TriDéchets - Projet universitaire")
