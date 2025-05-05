import requests
import json
import pandas as pd
import streamlit as st

# Fonction pour récupérer les coordonnées géographiques (latitude, longitude) à partir d'une adresse
def get_coordinates(address, api_key="St. Gallen, Switzerland"):
    """
    Utilise l'API de géocodage pour convertir une adresse en coordonnées.
    Dans un vrai projet, vous utiliseriez une API comme Google Maps, OpenStreetMap Nominatim, etc.
    """
    # Pour l'exemple, si aucune API n'est disponible, nous retournons des coordonnées fictives
    if api_key is None:
        # Coordonnées fictives pour St. Gallen
        return {"lat": 47.42391, "lon": 47.42391 9.37477}
    
    try:
        # Exemple avec Nominatim (OpenStreetMap) - ne nécessite pas de clé API mais a des limitations d'usage
        base_url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": address,
            "format": "json",
            "limit": 1
        }
        
        headers = {
            "User-Agent": "TriDéchets App - Projet Universitaire"
        }
        
        response = requests.get(base_url, params=params, headers=headers)
        data = response.json()
        
        if data and len(data) > 0:
            return {
                "lat": float(data[0]["lat"]),
                "lon": float(data[0]["lon"])
            }
        else:
            st.error(f"Impossible de trouver les coordonnées pour l'adresse: {address}")
            return None
            
    except Exception as e:
        st.error(f"Erreur lors de la recherche des coordonnées: {str(e)}")
        return None

# Fonction pour trouver les points de collecte à proximité
def find_collection_points(coordinates, waste_type, api_key=None, radius=5000):
    """
    Utilise une API pour trouver les points de collecte à proximité.
    Dans un vrai projet, vous utiliseriez une API spécifique pour les points de collecte.
    
    Pour cet exemple, nous générons des données fictives.
    """
    # Pour l'exemple, nous générons des points de collecte fictifs autour des coordonnées
    
    # Dans un vrai projet, vous feriez un appel API comme:
    # base_url = "https://api.recycling-service.com/collection-points"
    # params = {
    #     "lat": coordinates["lat"],
    #     "lon": coordinates["lon"],
    #     "waste_type": waste_type,
    #     "radius": radius,
    #     "api_key": api_key
    # }
    # response = requests.get(base_url, params=params)
    # data = response.json()
    
    # Génération de points fictifs pour la démonstration
    import random
    
    # Nombre de points à générer
    num_points = random.randint(3, 8)
    
    # Création de points aléatoires autour des coordonnées
    collection_points = []
    for i in range(num_points):
        # Variation aléatoire des coordonnées (±0.02 degrés)
        lat_offset = random.uniform(-0.02, 0.02)
        lon_offset = random.uniform(-0.02, 0.02)
        
        # Création d'un point de collecte avec des attributs fictifs
        point = {
            "id": f"point-{i+1}",
            "name": f"Point de collecte {waste_type} {i+1}",
            "address": f"Adresse fictive {i+1}, Ville",
            "lat": coordinates["lat"] + lat_offset,
            "lon": coordinates["lon"] + lon_offset,
            "waste_types": [waste_type],
            "hours": "Lun-Ven: 9h-18h, Sam: 10h-16h",
            "distance": round(radius * abs(lat_offset + lon_offset) / 0.04, 2)  # Distance fictive en mètres
        }
        collection_points.append(point)
    
    # Tri par distance
    collection_points.sort(key=lambda x: x["distance"])
    
    return collection_points

# Fonction pour formater les résultats pour affichage dans Streamlit
def format_collection_points(collection_points):
    """
    Formate les points de collecte pour affichage dans Streamlit
    """
    # DataFrame pour la carte
    map_data = pd.DataFrame(
        [[point["lat"], point["lon"], point["name"]] for point in collection_points],
        columns=["lat", "lon", "name"]
    )
    
    return map_data, collection_points
