import pickle
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import re

# Fonction pour charger le modèle, vectoriseur et encodeur
def load_model_and_transformers():
    try:
        with open('waste_classifier.pkl', 'rb') as f:
            model = pickle.load(f)
            
        with open('waste_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
            
        with open('waste_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
            
        return model, vectorizer, encoder
    except FileNotFoundError:
        return None, None, None

# Fonction pour prédire la catégorie à partir d'une description textuelle
def predict_from_text(description, model, vectorizer, encoder):
    """
    Prédit la catégorie de déchet à partir d'une description textuelle
    """
    if not description:
        return None, 0.0
    
    # Prétraitement du texte
    description = description.lower()
    
    # Vectorisation
    X_new = vectorizer.transform([description])
    
    # Prédiction
    prediction = model.predict(X_new)[0]
    probabilities = model.predict_proba(X_new)[0]
    confidence = probabilities[prediction]
    
    # Récupération de la catégorie
    category = encoder.inverse_transform([prediction])[0]
    
    return category, confidence

# Dictionnaire des mots-clés pour chaque catégorie (pour la méthode basée sur les règles)
WASTE_KEYWORDS = {
    "Plastique": ["plastique", "pet", "bouteille", "emballage", "polyéthylène", "polystyrène", "barquette", "sac"],
    "Papier/Carton": ["papier", "carton", "journal", "magazine", "livre", "enveloppe", "boîte", "emballage carton"],
    "Verre": ["verre", "bouteille en verre", "pot en verre", "vitre", "miroir", "bocal"],
    "Métal": ["métal", "aluminium", "acier", "fer", "canette", "conserve", "boîte métallique"],
    "Électronique": ["électronique", "électrique", "pile", "batterie", "téléphone", "ordinateur", "câble", "chargeur"],
    "Textile": ["textile", "vêtement", "tissu", "chaussure", "coton", "laine", "cuir"],
    "Dangereux": ["toxique", "peinture", "solvant", "huile", "produit chimique", "ampoule", "néon", "médicament"],
    "Organique": ["organique", "alimentaire", "compost", "fruit", "légume", "épluchure", "jardin", "végétal"]
}

# Fonction de prédiction basée sur les règles (utilisée comme fallback si le ML n'est pas disponible)
def rule_based_prediction(description):
    """
    Prédit la catégorie de déchet basée sur des règles simples de correspondance de mots-clés
    """
    description = description.lower()
    
    # Scores pour chaque catégorie
    scores = {}
    
    for category, keywords in WASTE_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in description:
                score += 1
        
        scores[category] = score
    
    # Trouver la catégorie avec le score le plus élevé
    if any(scores.values()):  # Si au moins un mot-clé a été trouvé
        best_category = max(scores, key=scores.get)
        confidence = scores[best_category] / len(WASTE_KEYWORDS[best_category])
        return best_category, min(confidence, 0.9)  # Limiter la confiance à 0.9 max pour cette méthode simple
    else:
        return "Autre", 0.3  # Catégorie par défaut si aucun mot-clé ne correspond

# Fonction pour obtenir des conseils de tri pour une catégorie de déchet
def get_waste_tips(category):
    """
    Retourne des conseils de tri pour une catégorie spécifique de déchet
    """
    tips = {
        "Plastique": {
            "poubelle": "Poubelle jaune (recyclage)",
            "conseils": [
                "Videz et rincez légèrement avant de jeter",
                "Enlevez les bouchons et jetez-les séparément dans la poubelle jaune",
                "Certains plastiques ne sont pas recyclables (polystyrène, films plastiques fins)"
            ]
        },
        "Papier/Carton": {
            "poubelle": "Poubelle bleue (recyclage)",
            "conseils": [
                "Pliez les cartons pour économiser de l'espace",
                "Retirez les éléments non-papier (scotch, agrafes)",
                "Le papier souillé (gras, alimentaire) va dans la poubelle ordinaire"
            ]
        },
        "Verre": {
            "poubelle": "Conteneur à verre",
            "conseils": [
                "Videz bien les contenants mais ne les rincez pas",
                "Retirez les bouchons et couvercles",
                "La vaisselle, vitres et miroirs ne vont pas dans le conteneur à verre"
            ]
        },
        "Métal": {
            "poubelle": "Poubelle jaune (recyclage)",
            "conseils": [
                "Videz et rincez légèrement",
                "Les petits éléments métalliques peuvent être collectés dans une boîte"
            ]
        },
        "Électronique": {
            "poubelle": "Déchetterie ou point de collecte spécifique",
            "conseils": [
                "Rapportez vos appareils en magasin lors d'un nouvel achat",
                "Certaines associations récupèrent les appareils fonctionnels",
                "Les piles ont des points de collecte dédiés en magasin"
            ]
        },
        "Textile": {
            "poubelle": "Borne de collecte textile",
            "conseils": [
                "Mettez les textiles propres dans un sac fermé",
                "Les vêtements en bon état peuvent être donnés à des associations",
                "Les textiles très usés ou déchirés sont aussi acceptés"
            ]
        },
        "Dangereux": {
            "poubelle": "Déchetterie ou point de collecte spécifique",
            "conseils": [
                "Ne jamais jeter dans les poubelles ordinaires ou les canalisations",
                "Conservez dans le contenant d'origine si possible",
                "Les médicaments doivent être rapportés en pharmacie"
            ]
        },
        "Organique": {
            "poubelle": "Composteur ou poubelle de déchets organiques",
            "conseils": [
                "Idéal pour un composteur individuel ou collectif",
                "Évitez les aliments cuits avec de la viande ou du poisson",
                "Mélangez avec des déchets secs (feuilles, carton)"
            ]
        },
        "Autre": {
            "poubelle": "Poubelle ordinaire (déchets non recyclables)",
            "conseils": [
                "Vérifiez s'il existe une filière de recyclage spécifique",
                "Privilégiez les déchetteries pour les objets volumineux",
                "Pensez au réemploi ou au don si l'objet est encore utilisable"
            ]
        }
    }
    
    return tips.get(category, tips["Autre"])

# Fonction pour une prédiction simplifiée à partir d'une image
def simple_image_prediction(image):
    """
    Simule une prédiction à partir d'une image.
    Dans un vrai projet, vous utiliseriez un modèle de vision par ordinateur.
    Cette fonction est simplement un placeholder.
    """
    # Convertir l'image en tableau numpy
    img_array = np.array(image)
    
    # Analyse simplifiée des couleurs (juste pour la démonstration)
    # Dans un projet réel, vous utiliseriez un modèle CNN comme ResNet, EfficientNet, etc.
    avg_color = np.mean(img_array, axis=(0, 1))
    
    # Logic très basique basée sur la couleur moyenne
    # Ceci est juste pour démonstration et ne fonctionnera pas bien en pratique
    if avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:  # Si dominante verte
        return "Organique", 0.6
    elif avg_color[2] > avg_color[0] and avg_color[2] > avg_color[1]:  # Si dominante bleue
        return "Papier/Carton", 0.5
    elif avg_color[0] > 200 and avg_color[1] > 200 and avg_color[2] > 200:  # Si très clair
        return "Plastique", 0.4
    elif avg_color[0] < 50 and avg_color[1] < 50 and avg_color[2] < 50:  # Si très sombre
        return "Métal", 0.4
    else:
        return "Autre", 0.3
