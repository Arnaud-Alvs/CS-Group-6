import requests
import json
import pandas as pd
import streamlit as st
from datetime import datetime
import time
from typing import Dict, List, Optional, Any, Tuple

# Base URL pour l'API
BASE_API_URL = "https://data.stadt.sg.ch"

# Fonction pour récupérer les coordonnées géographiques (latitude, longitude) à partir d'une adresse
def get_coordinates(address: str, api_key: Optional[str] = None) -> Optional[Dict[str, float]]:
    """
    Utilise l'API de géocodage pour convertir une adresse en coordonnées.
    
    Args:
        address: L'adresse à géocoder
        api_key: Clé API optionnelle pour les services payants
        
    Returns:
        Un dictionnaire contenant les clés 'lat' et 'lon', ou None en cas d'erreur
    """
    try:
        # Utilisation de Nominatim (OpenStreetMap) - ne nécessite pas de clé API mais a des limitations d'usage
        base_url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": address,
            "format": "json",
            "limit": 1,
            "country": "Switzerland"  # Limitation à la Suisse pour de meilleurs résultats
        }
        
        headers = {
            "User-Agent": "TriDéchets App - Projet Universitaire"
        }
        
        # Ajout d'un délai pour respecter les conditions d'utilisation de Nominatim
        time.sleep(1)
        
        response = requests.get(base_url, params=params, headers=headers)
        
        if response.status_code != 200:
            st.error(f"Erreur de l'API de géocodage: {response.status_code}")
            return None
            
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

# Fonction pour récupérer les points de collecte réels depuis l'API
def get_collection_points_api(waste_type: str) -> List[Dict[str, Any]]:
    """
    Récupère tous les points de collecte depuis l'API
    
    Args:
        waste_type: Type de déchet recherché
        
    Returns:
        Liste de points de collecte
    """
    try:
        api_url = f"{BASE_API_URL}/api/explore/v2.1/catalog/datasets/sammelstellen/records"
        params = {
            "limit": 100  # Augmenter la limite pour récupérer plus de points
        }
        
        response = requests.get(api_url, params=params)
        
        if response.status_code != 200:
            st.error(f"Erreur de l'API des points de collecte: {response.status_code}")
            return []
            
        data = response.json()
        
        collection_points = []
        
        for point in data.get("results", []):
            # Vérifier si le type de déchet est accepté à ce point de collecte
            if "abfallarten" in point and waste_type in point["abfallarten"]:
                # Extraire les coordonnées
                geo = point.get("geo_point_2d", {})
                
                collection_point = {
                    "id": point.get("sammelstel", ""),
                    "name": f"Point de collecte {point.get('sammelstel', '')}",
                    "address": point.get("standort", "Adresse non spécifiée"),
                    "lat": geo.get("lat", 0),
                    "lon": geo.get("lon", 0),
                    "waste_types": point.get("abfallarten", []),
                    "hours": point.get("oeffnungsz", "Horaires non spécifiés"),
                    "distance": 0  # Sera calculé plus tard
                }
                collection_points.append(collection_point)
        
        return collection_points
        
    except Exception as e:
        st.error(f"Erreur lors de la récupération des points de collecte: {str(e)}")
        return []

# Fonction pour calculer la distance entre deux points géographiques
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcule la distance approximative en kilomètres entre deux points géographiques
    en utilisant la formule de Haversine
    
    Args:
        lat1, lon1: Coordonnées du premier point
        lat2, lon2: Coordonnées du deuxième point
        
    Returns:
        Distance en kilomètres
    """
    from math import radians, sin, cos, sqrt, atan2
    
    # Conversion en radians
    lat1, lon1 = radians(lat1), radians(lon1)
    lat2, lon2 = radians(lat2), radians(lon2)
    
    # Rayon de la Terre en km
    R = 6371.0
    
    # Différences
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    # Formule de Haversine
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    
    return distance

# Fonction principale pour trouver les points de collecte à proximité
def find_collection_points(coordinates: Dict[str, float], waste_type: str, radius: float = 5.0) -> List[Dict[str, Any]]:
    """
    Trouve les points de collecte à proximité pour un type de déchet spécifique
    
    Args:
        coordinates: Dictionnaire contenant la latitude et longitude du point de départ
        waste_type: Type de déchet recherché
        radius: Rayon de recherche en kilomètres
        
    Returns:
        Liste des points de collecte triés par distance
    """
    if not coordinates:
        return []
    
    # Récupérer tous les points de collecte
    all_points = get_collection_points_api(waste_type)
    
    # Calculer la distance pour chaque point et filtrer par rayon
    nearby_points = []
    
    for point in all_points:
        distance = calculate_distance(
            coordinates["lat"], coordinates["lon"],
            point["lat"], point["lon"]
        )
        
        # Convertir en kilomètres et arrondir à 2 décimales
        point["distance"] = round(distance, 2)
        
        # Filtrer par rayon
        if distance <= radius:
            nearby_points.append(point)
    
    # Trier par distance
    nearby_points.sort(key=lambda x: x["distance"])
    
    return nearby_points

# Fonction pour récupérer les dates de collecte
def get_collection_dates(waste_type: str, address: str) -> List[Dict[str, Any]]:
    """
    Récupère les prochaines dates de collecte pour un type de déchet et une adresse
    
    Args:
        waste_type: Type de déchet (ex: "Kehricht", "Karton", "Papier", etc.)
        address: Adresse pour laquelle rechercher les dates
        
    Returns:
        Liste des prochaines dates de collecte
    """
    try:
        # Extraire le nom de la rue de l'adresse
        # Format attendu: "Rue, Numéro, Ville"
        address_parts = address.split(',')
        if len(address_parts) < 1:
            return []
        
        street_name = address_parts[0].strip()
        
        # Récupérer l'année courante
        current_year = datetime.now().year
        
        # URL de l'API pour les dates de collecte
        api_url = f"{BASE_API_URL}/api/explore/v2.1/catalog/datasets/abfuhrdaten-stadt-stgallen/records"
        
        params = {
            "limit": 100,
            "refine.datum": str(current_year)  # Filtrer par année courante
        }
        
        response = requests.get(api_url, params=params)
        
        if response.status_code != 200:
            st.error(f"Erreur de l'API des dates de collecte: {response.status_code}")
            return []
            
        data = response.json()
        
        # Conversion de waste_type en termes utilisés par l'API
        waste_type_mapping = {
            "Aluminium": "Alu+Weissblech",
            "Dosen": "Alu+Weissblech",
            "Glas": "Glas",
            "Papier": "Papier",
            "Karton": "Karton",
            "Kehricht": "Kehricht",  # Ordures ménagères
            "Metall": "Metall",
            "Sonderabfall": "Sonderabfall",  # Déchets spéciaux
            "Grüngut": "Grüngut"  # Déchets verts
        }
        
        api_waste_type = waste_type_mapping.get(waste_type, waste_type)
        
        # Filtrer les résultats
        collection_dates = []
        
        for entry in data.get("results", []):
            # Vérifier si le type de collecte correspond
            if entry.get("sammlung") == api_waste_type:
                # Vérifier si la rue est dans la liste des rues concernées
                streets = entry.get("strasse", [])
                
                # Recherche partielle du nom de la rue
                street_match = any(street.lower() in street_name.lower() or street_name.lower() in street.lower() for street in streets)
                
                if street_match:
                    # Formater la date
                    date_str = entry.get("datum", "")
                    time_str = entry.get("zeit", "")
                    
                    try:
                        collection_date = datetime.strptime(date_str, "%Y-%m-%d")
                        
                        # Ne conserver que les dates futures
                        if collection_date >= datetime.now():
                            collection_info = {
                                "date": date_str,
                                "time": time_str,
                                "area": entry.get("gebietsbezeichnung", ""),
                                "title": entry.get("titel", ""),
                                "pdf": entry.get("pdf", "")
                            }
                            collection_dates.append(collection_info)
                    except ValueError:
                        continue
        
        # Trier par date
        collection_dates.sort(key=lambda x: x["date"])
        
        return collection_dates
        
    except Exception as e:
        st.error(f"Erreur lors de la récupération des dates de collecte: {str(e)}")
        return []

# Fonction pour formater les résultats pour affichage dans Streamlit
def format_collection_points(collection_points: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Formate les points de collecte pour affichage dans Streamlit
    
    Args:
        collection_points: Liste des points de collecte
        
    Returns:
        Tuple contenant (DataFrame pour la carte, Liste originale des points)
    """
    if not collection_points:
        return pd.DataFrame(columns=["lat", "lon", "name"]), []
    
    # DataFrame pour la carte
    map_data = pd.DataFrame(
        [[point["lat"], point["lon"], point["name"]] for point in collection_points],
        columns=["lat", "lon", "name"]
    )
    
    return map_data, collection_points

# Fonction utilitaire pour traduire les types de déchets
def translate_waste_type(waste_type: str) -> str:
    """
    Traduit les types de déchets de l'API en français
    
    Args:
        waste_type: Type de déchet en allemand
        
    Returns:
        Type de déchet traduit en français
    """
    translations = {
        "Aluminium": "Aluminium",
        "Dosen": "Boîtes de conserve",
        "Glas": "Verre",
        "Papier": "Papier",
        "Karton": "Carton",
        "Kehricht": "Ordures ménagères",
        "Metall": "Métal",
        "Sonderabfall": "Déchets spéciaux",
        "Grüngut": "Déchets verts"
    }
    
    return translations.get(waste_type, waste_type)

# Fonction pour obtenir la liste de tous les types de déchets disponibles
def get_available_waste_types() -> List[str]:
    """
    Récupère la liste de tous les types de déchets disponibles dans l'API
    
    Returns:
        Liste des types de déchets
    """
    try:
        api_url = f"{BASE_API_URL}/api/explore/v2.1/catalog/datasets/sammelstellen/records"
        params = {
            "limit": 100
        }
        
        response = requests.get(api_url, params=params)
        
        if response.status_code != 200:
            return ["Aluminium", "Dosen", "Glas", "Papier", "Karton", "Kehricht"]
            
        data = response.json()
        
        # Extraire tous les types de déchets uniques
        waste_types = set()
        
        for point in data.get("results", []):
            if "abfallarten" in point:
                waste_types.update(point["abfallarten"])
        
        return sorted(list(waste_types))
        
    except Exception:
        # En cas d'erreur, retourner une liste par défaut
        return ["Aluminium", "Dosen", "Glas", "Papier", "Karton", "Kehricht"]
