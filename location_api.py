import requests
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from math import radians, sin, cos, sqrt, atan2

# Base API URLs
BASE_API_URL = "https://daten.stadt.sg.ch"
COLLECTION_POINTS_ENDPOINT = f"{BASE_API_URL}/api/explore/v2.1/catalog/datasets/sammelstellen/records"
COLLECTION_DATES_ENDPOINT = f"{BASE_API_URL}/api/explore/v2.1/catalog/datasets/abfuhrdaten-stadt-stgallen/records"

# ----------------------------------------
# GET COORDINATES
# ----------------------------------------

def get_coordinates(address: str, api_key: Optional[str] = None) -> Optional[Dict[str, float]]:
    """
    Get latitude and longitude from address using OpenStreetMap Nominatim API.
    """
    try:
        base_url = "https://nominatim.openstreetmap.org/search"

        if "st.gallen" not in address.lower() and "st. gallen" not in address.lower():
            address = f"{address}, St. Gallen, Switzerland"

        params = {
            "q": address,
            "format": "json",
            "limit": 1,
            "addressdetails": 1,
            "countrycodes": "ch",
        }

        headers = {
            "User-Agent": "WasteWise App - University Project"
        }

        response = requests.get(base_url, params=params, headers=headers, timeout=30)

        if response.status_code != 200:
            st.error(f"Geocoding API error: {response.status_code}")
            return None

        data = response.json()

        if not data:
            st.error(f"Address not found: {address}")
            return None

        result = data[0]
        address_details = result.get("address", {})

        possible_location_keys = ["city", "town", "village", "municipality", "county", "state"]
        location_found = any("st. gallen" in address_details.get(key, "").lower() or 
                             "st.gallen" in address_details.get(key, "").lower() 
                             for key in possible_location_keys)

        if location_found:
            return {"lat": float(result["lat"]), "lon": float(result["lon"])}
        else:
            st.warning(f"The address was found but appears outside St. Gallen. Please verify.")
            return None

    except Exception as e:
        st.error(f"Error searching for coordinates: {str(e)}")
        return None

# ----------------------------------------
# COLLECTION POINTS API
# ----------------------------------------

def get_collection_points_api(waste_type: str) -> List[Dict[str, Any]]:
    try:
        params = {"limit": 100}
        headers = {"User-Agent": "WasteWise App", "Accept": "application/json"}

        response = requests.get(COLLECTION_POINTS_ENDPOINT, params=params, headers=headers, timeout=30)
        if response.status_code != 200:
            st.error("Collection points API error")
            return get_fallback_collection_points(waste_type)

        data = response.json()
        results = data.get("results", [])
        collection_points = []

        for result in results:
            fields = result.get("record", {}).get("fields", {})
            waste_types = fields.get("abfallarten", [])
            if isinstance(waste_types, str):
                waste_types = [w.strip() for w in waste_types.split(",")]

            if waste_type not in waste_types and not any(waste_type.lower() in wt.lower() for wt in waste_types):
                continue

            geo_point = fields.get("geo_point_2d", {})
            lat, lon = None, None
            if isinstance(geo_point, dict):
                lat = geo_point.get("lat")
                lon = geo_point.get("lon")
            elif isinstance(geo_point, list) and len(geo_point) >= 2:
                lat, lon = geo_point

            if lat and lon:
                collection_points.append({
                    "id": fields.get("sammelstel", ""),
                    "name": f"Collection Point {fields.get('sammelstel', '')}",
                    "address": fields.get("standort", "Address not specified"),
                    "lat": float(lat),
                    "lon": float(lon),
                    "waste_types": waste_types,
                    "hours": fields.get("oeffnungsz", "Hours not specified"),
                    "distance": 0
                })

        return collection_points or get_fallback_collection_points(waste_type)

    except Exception as e:
        st.error(f"Error retrieving collection points: {str(e)}")
        return get_fallback_collection_points(waste_type)

def get_fallback_collection_points(waste_type: str) -> List[Dict[str, Any]]:
    fallback_points = [
        {
            "id": "fallback_01",
            "name": "St. Gallen Main Recycling Center",
            "address": "Kehrichtverbrennungsanlage KVA, Hüettenwiesstrasse 50, 9014 St. Gallen",
            "lat": 47.4245,
            "lon": 9.3767,
            "waste_types": ["Kehricht", "Papier", "Karton", "Glas", "Altmetall", "Sonderabfall",
                            "Alttextilien", "Altöl", "Styropor", "Grüngut", "Dosen", "Aluminium"],
            "hours": "Monday-Friday 8:00-17:00, Saturday 8:00-12:00",
            "distance": 0
        }
    ]
    return [p for p in fallback_points if waste_type in p["waste_types"]] or fallback_points

# ----------------------------------------
# DISTANCE CALCULATION
# ----------------------------------------

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return R * (2 * atan2(sqrt(a), sqrt(1 - a)))

def find_collection_points(coordinates: Dict[str, float], waste_type: str, radius: float = 10.0) -> List[Dict[str, Any]]:
    if not coordinates:
        return []
    
    all_points = get_collection_points_api(waste_type)
    nearby_points = []

    for point in all_points:
        distance = calculate_distance(coordinates["lat"], coordinates["lon"], point["lat"], point["lon"])
        point["distance"] = round(distance, 2)
        if distance <= radius:
            nearby_points.append(point)

    return sorted(nearby_points, key=lambda x: x["distance"])

# ----------------------------------------
# COLLECTION DATES API
# ----------------------------------------

def get_collection_dates(waste_type: str, address: str) -> List[Dict[str, Any]]:
    try:
        street_name = address.split(",")[0].strip()
        current_year = datetime.now().year

        params = {"limit": 20, "refine.datum": str(current_year)}
        headers = {"User-Agent": "WasteWise App", "Accept": "application/json"}

        response = requests.get(COLLECTION_DATES_ENDPOINT, params=params, headers=headers, timeout=30)
        if response.status_code != 200:
            return get_fallback_collection_dates(waste_type)

        data = response.json()
        collection_dates = []
        results = data.get("results", [])

        waste_type_mapping = {
            "Aluminium": "Alu+Weissblech", "Dosen": "Alu+Weissblech", "Glas": "Glas", "Papier": "Papier", "Karton": "Karton",
            "Kehricht": "Kehricht", "Altmetall": "Metall", "Sonderabfall": "Sonderabfall", "Grüngut": "Grüngut",
            "Alttextilien": "Alttextilien", "Altöl": "Altöl", "Styropor": "Styropor"
        }
        api_waste_type = waste_type_mapping.get(waste_type, waste_type)

        for result in results:
            fields = result.get("record", {}).get("fields", {})
            if fields.get("sammlung") == api_waste_type:
                streets = fields.get("strasse", [])
                if isinstance(streets, str):
                    streets = [streets]

                if any(street.lower() in street_name.lower() for street in streets):
                    collection_date = datetime.strptime(fields.get("datum", ""), "%Y-%m-%d")
                    if collection_date >= datetime.now():
                        collection_dates.append({
                            "date": fields.get("datum", ""),
                            "time": fields.get("zeit", ""),
                            "area": fields.get("gebietsbezeichnung", ""),
                            "title": fields.get("titel", ""),
                            "pdf": fields.get("pdf", "")
                        })

        return sorted(collection_dates, key=lambda x: x["date"]) or get_fallback_collection_dates(waste_type)

    except Exception as e:
        st.error(f"Error retrieving collection dates: {str(e)}")
        return get_fallback_collection_dates(waste_type)

def get_fallback_collection_dates(waste_type: str) -> List[Dict[str, Any]]:
    today = datetime.now()
    collection_schedules = {
        "Kehricht": [7, 14, 21, 28], "Papier": [10, 24], "Karton": [10, 24], "Grüngut": [3, 17, 31],
        "Sonderabfall": [15], "Glas": [20], "Dosen": [10, 24], "Aluminium": [10, 24], "Altmetall": [15],
        "Alttextilien": [15], "Altöl": [15], "Styropor": [15]
    }
    
    return [{
        "date": datetime(today.year, today.month, day).strftime("%Y-%m-%d"),
        "time": "07:00",
        "area": "St. Gallen Center",
        "title": f"{translate_waste_type(waste_type)} Collection",
        "pdf": None
    } for day in collection_schedules.get(waste_type, [15]) if datetime(today.year, today.month, day) >= today]

# ----------------------------------------
# FORMAT COLLECTION POINTS FOR STREAMLIT
# ----------------------------------------

def format_collection_points(collection_points: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    if not collection_points:
        return pd.DataFrame(columns=["lat", "lon", "name"]), []

    map_data = pd.DataFrame([[p["lat"], p["lon"], p["name"]] for p in collection_points], columns=["lat", "lon", "name"])
    return map_data, collection_points

# ----------------------------------------
# TRANSLATION
# ----------------------------------------

def translate_waste_type(waste_type: str) -> str:
    translations = {
        "Aluminium": "Aluminium", "Dosen": "Cans", "Glas": "Glass", "Papier": "Paper", "Karton": "Cardboard",
        "Kehricht": "Household waste", "Altmetall": "Metal", "Sonderabfall": "Hazardous waste", "Grüngut": "Green waste",
        "Alttextilien": "Textiles", "Altöl": "Oil", "Styropor": "Foam packaging"
    }
    return translations.get(waste_type, waste_type)

def get_available_waste_types() -> List[str]:
    return ["Kehricht", "Papier", "Karton", "Glas", "Grüngut", "Dosen", "Aluminium",
            "Altmetall", "Alttextilien", "Altöl", "Sonderabfall", "Styropor"]

