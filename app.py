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
    page_title="TriDéchets - Your smart recycling assistant",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre de l'application
st.title("TriDéchets - Your smart recycling assistant")
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
        # Pour l'instant, nous affichons une réponse fictive
        if waste_description or uploaded_file:
            st.success("D'après notre analyse, il s'agit probablement d'un déchet recyclable de type PLASTIQUE")
            st.info("Conseils de tri : Ce déchet doit être jeté dans la poubelle jaune pour le recyclage")
        else:
            st.error("Veuillez décrire votre déchet ou télécharger une image")

with tab3:
    st.header("À propos de TriDéchets")
    
    # Description du projet
    st.markdown("""
    ## Notre mission
    
    **TriDéchets** est une application éducative développée dans le cadre d'un projet universitaire. 
    Notre mission est de simplifier le tri des déchets et de contribuer à un meilleur recyclage en:
    
    - Aidant les utilisateurs à identifier correctement leurs déchets
    - Fournissant des conseils personnalisés pour le tri
    - Localisant les points de collecte les plus proches
    - Sensibilisant à l'importance du recyclage
    
    ## Technologies utilisées
    
    Cette application est construite avec:
    
    - **Streamlit**: pour l'interface utilisateur
    - **Machine Learning**: pour l'identification des déchets
    - **API de géolocalisation**: pour trouver les points de collecte
    - **Traitement de données**: pour analyser et classifier les déchets
    
    ## Comment ça marche?
    
    1. **Identification de déchets**: Notre système d'IA analyse votre description ou votre photo pour déterminer le type de déchet.
    
    2. **Conseils personnalisés**: En fonction du type de déchet identifié, nous vous fournissons des conseils spécifiques pour son tri.
    
    3. **Localisation des points de collecte**: Nous vous aidons à trouver les points de collecte les plus proches de chez vous pour déposer vos déchets.
    """)
    
    # Affichage de statistiques fictives sur l'utilisation de l'application
    st.subheader("Impact de TriDéchets")
    
    # Utilisation de colonnes pour afficher des statistiques côte à côte
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Déchets identifiés", value="12,543", delta="1,243 cette semaine")
    
    with col2:
        st.metric(label="Points de collecte", value="3,879", delta="152 nouveaux")
    
    with col3:
        st.metric(label="Utilisateurs actifs", value="853", delta="57 nouveaux")
    
    # Feedback utilisateur
    st.subheader("Votre avis compte")
    st.write("Aidez-nous à améliorer TriDéchets en nous faisant part de vos suggestions:")
    
    with st.form("feedback_form"):
        feedback_text = st.text_area("Vos commentaires et suggestions")
        user_email = st.text_input("Votre email (facultatif)")
        submit_button = st.form_submit_button("Envoyer mon feedback")
        
        if submit_button:
            if feedback_text:
                st.success("Merci pour votre feedback! Nous l'avons bien reçu.")
                # Dans un vrai projet, vous enregistreriez ce feedback dans une base de données
            else:
                st.error("Veuillez entrer un commentaire avant d'envoyer.")

# Barre latérale avec des fonctionnalités supplémentaires
with st.sidebar:
    st.header("Options")
    
    # Mode sombre/clair
    if st.checkbox("Mode sombre", value=False):
        st.markdown("""
        <style>
        .stApp {
            background-color: #121212;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Langue de l'application (simulée)
    st.selectbox("Langue", ["Français", "English", "Español", "Deutsch"])
    
    # Conseils du jour
    st.subheader("Le conseil du jour")
    tips_of_the_day = [
        "Savez-vous que les bouchons en plastique peuvent être recyclés séparément?",
        "Le verre se recycle à l'infini sans perdre sa qualité!",
        "Un téléphone portable contient plus de 70 matériaux différents, dont beaucoup sont recyclables.",
        "Les piles contiennent des métaux lourds toxiques, ne les jetez jamais avec les ordures ménagères.",
        "Pensez à apposer un autocollant 'Stop Pub' sur votre boîte aux lettres pour réduire vos déchets papier."
    ]
    import random
    st.info(random.choice(tips_of_the_day))
    
    # Séparateur
    st.markdown("---")
    
    # Liens utiles
    st.subheader("Liens utiles")
    st.markdown("[Guide complet du recyclage](https://example.com)")
    st.markdown("[Réduire ses déchets au quotidien](https://example.com)")
    st.markdown("[Législation sur les déchets](https://example.com)")

# Pied de page
st.markdown("---")
st.markdown("© 2025 TriDéchets - Projet universitaire | [Contact](mailto:contact@tridechets.example.com) | [Mentions légales](https://example.com)")

# Pied de page
st.markdown("---")
st.markdown("© 2025 TriDéchets - Projet universitaire")
