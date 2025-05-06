import requests
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from math import radians, sin, cos, sqrt, atan2

# Base API URLs
BASE_API_URL = "https://daten.stadt.sg.ch"

# Function to get geographic coordinates (latitude, longitude) from an address
def get_coordinates(address: str, api_key: Optional[str] = None) -> Optional[Dict[str, float]]:
    """
    Get latitude and longitude from address using OpenStreetMap Nominatim API.
    """
    try:
        base_url = "https://nominatim.openstreetmap.org/search"

        if "st.gallen" not in address.lower() and "st. gallen" not in address.lower():
            if "switzerland" not in address.lower() and "schweiz" not in address.lower():
                address = f"{address}, St. Gallen, Switzerland"
            else:
                address = address.replace("Switzerland", "St. Gallen, Switzerland").replace("schweiz", "St. Gallen, Schweiz")
            
        params = {
            "q": address,
            "format": "json",
            "limit": 1,
            "addressdetails": 1,
            "countrycodes": "ch",  # Limit to Switzerland
            "city": "St. Gallen"   # Focus on St. Gallen
        }

        headers = {
            "User-Agent": "WasteWise App - University Project"
        }

        response = requests.get(base_url, params=params, headers=headers, timeout=30)

        if response.status_code != 200:
            st.error(f"Geocoding API error: {response.status_code}")
            return None

        data = response.json()
        
        if data and len(data) > 0:
            # Verify the result is in St. Gallen
            result = data[0]
            address_details = result.get("address", {})
            city = address_details.get("city", "").lower()
            municipality = address_details.get("municipality", "").lower()
            
            if "st. gallen" in city or "st.gallen" in city or "st. gallen" in municipality or "st.gallen" in municipality:
                return {
                    "lat": float(result["lat"]),
                    "lon": float(result["lon"])
                }
            else:
                st.warning(f"The address was found outside St. Gallen. Please enter an address within St. Gallen.")
                return None
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
    """
    Retrieves all collection points from the API
    
    Args:
        waste_type: Type of waste being searched for
        
    Returns:
        List of collection points
    """
    try:
        api_url = f"{BASE_API_URL}/api/explore/v2.1/catalog/datasets/sammelstellen/records"
        params = {
            "limit": 100  # Get more results
        }
        
        response = requests.get(api_url, params=params, timeout=30)
        
        if response.status_code != 200:
            st.error(f"Collection points API error: {response.status_code}")
            return []
            
        data = response.json()
        results = data.get("results", [])
        collection_points = []
        
        for entry in data.get("results", []):
            # Extract fields - the API structure might be different
            fields = entry.get("fields", {}) or entry.get("record", {}).get("fields", {})
            
            # Get waste types - try different field names
            waste_types = fields.get("abfallarten", [])
            if not waste_types:
                waste_types = fields.get("abfalltyp", [])
            if not waste_types:
                waste_types = fields.get("art", [])
            
            # Handle case where waste_types is a string
            if isinstance(waste_types, str):
                waste_types = [w.strip() for w in waste_types.split(',')]
            
            # Check if the waste type is accepted
            if waste_type in waste_types or any(waste_type.lower() in wt.lower() for wt in waste_types):
                # Get coordinates - try different structures
                lat = lon = None
                
                # Try various possible locations for coordinates
                if "geo_point_2d" in fields:
                    geo = fields["geo_point_2d"]
                    if isinstance(geo, dict):
                        lat = geo.get("lat")
                        lon = geo.get("lon")
                    elif isinstance(geo, list) and len(geo) >= 2:
                        lat, lon = geo[0], geo[1]
                
                if not lat or not lon:
                    if "geometry" in entry:
                        geometry = entry["geometry"]
                        coordinates = geometry.get("coordinates", [])
                        if len(coordinates) >= 2:
                            lon, lat = coordinates[0], coordinates[1]  # GeoJSON format
                
                if not lat or not lon:
                    # Try other field names
                    lat = fields.get("lat") or fields.get("latitude")
                    lon = fields.get("lon") or fields.get("longitude")
                
                # Only add if we have valid coordinates
                if lat and lon:
                    # Get other information
                    sammelstel = fields.get("sammelstel", "")
                    standort = fields.get("standort", "Address not specified")
                    oeffnungsz = fields.get("oeffnungsz", "Hours not specified")
                    
                    collection_point = {
                        "id": sammelstel,
                        "name": f"Collection Point {sammelstel}" if sammelstel else "Collection Point",
                        "address": standort,
                        "lat": float(lat),
                        "lon": float(lon),
                        "waste_types": waste_types,
                        "hours": oeffnungsz,
                        "distance": 0
                    }
                    collection_points.append(collection_point)
        
        return collection_points
        
    except Exception as e:
        st.error(f"Error retrieving collection points: {str(e)}")
        return []

# Function to calculate distance between two geographic points
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
        
        # API URL for collection dates
        api_url = f"{BASE_API_URL}/api/explore/v2.1/catalog/datasets/abfuhrdaten-stadt-stgallen/records"
        
        params = {
            "limit": 100,
            "refine.datum": str(current_year)
        }
        
        response = requests.get(api_url, params=params, timeout=30)
        
        if response.status_code != 200:
            st.error(f"Collection dates API error: {response.status_code}")
            return []
            
        data = response.json()
        collection_dates = []
        results = data.get("results", [])

        waste_type_mapping = {
            "Aluminium": "Alu+Weissblech", "Dosen": "Alu+Weissblech", "Glas": "Glas", "Papier": "Papier", "Karton": "Karton",
            "Kehricht": "Kehricht", "Altmetall": "Metall", "Sonderabfall": "Sonderabfall", "Grüngut": "Grüngut",
            "Alttextilien": "Alttextilien", "Altöl": "Altöl", "Styropor": "Styropor"
        }
        api_waste_type = waste_type_mapping.get(waste_type, waste_type)
        
        # Filter results
        collection_dates = []
        
        for entry in data.get("results", []):
            fields = entry.get("fields", {}) or entry.get("record", {}).get("fields", {})
            
            # Check if collection type matches
            if fields.get("sammlung") == api_waste_type:
                streets = fields.get("strasse", [])
                if isinstance(streets, str):
                    streets = [streets]
                
                # Partial street name search
                street_match = any(street.lower() in street_name.lower() or street_name.lower() in street.lower() for street in streets)
                
                if street_match:
                    # Format date
                    date_str = fields.get("datum", "")
                    time_str = fields.get("zeit", "")
                    
                    try:
                        collection_date = datetime.strptime(date_str, "%Y-%m-%d")
                        
                        # Keep only future dates
                        if collection_date >= datetime.now():
                            collection_info = {
                                "date": date_str,
                                "time": time_str,
                                "area": fields.get("gebietsbezeichnung", ""),
                                "title": fields.get("titel", ""),
                                "pdf": fields.get("pdf", "")
                            }
                            collection_dates.append(collection_info)
                    except ValueError:
                        continue
        
        # Sort by date
        collection_dates.sort(key=lambda x: x["date"])
        
        return collection_dates
        
    except Exception as e:
        st.error(f"Error retrieving collection dates: {str(e)}")
        return []

# Function to format results for display in Streamlit
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

