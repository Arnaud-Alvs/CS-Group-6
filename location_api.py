import requests
import json
import pandas as pd
import streamlit as st
from datetime import datetime
import time
from typing import Dict, List, Optional, Any, Tuple
import urllib.parse

# Base URL for the API
BASE_API_URL = "https://daten.stadt.sg.ch"

# Function to get geographic coordinates (latitude, longitude) from an address
def get_coordinates(address: str, api_key: Optional[str] = None) -> Optional[Dict[str, float]]:
    """
    Uses geocoding API to convert an address to coordinates.
    
    Args:
        address: The address to geocode
        api_key: Optional API key for paid services
        
    Returns:
        Dictionary containing 'lat' and 'lon' keys, or None in case of error
    """
    try:
        # Using Nominatim (OpenStreetMap) - doesn't require an API key but has usage limitations
        base_url = "https://nominatim.openstreetmap.org/search"
        
        # Ensure address includes St. Gallen for better results
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
            "User-Agent": "WasteWise App - University Project (contact@example.com)"
        }
        
        # Add a delay to respect Nominatim's usage conditions
        time.sleep(1)
        
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
            st.error(f"Unable to find coordinates for address: {address}")
            return None
            
    except Exception as e:
        st.error(f"Error while searching for coordinates: {str(e)}")
        return None

# Function to retrieve real collection points from the API
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
    """
    Calculates the approximate distance in kilometers between two geographic points
    using the Haversine formula
    
    Args:
        lat1, lon1: Coordinates of the first point
        lat2, lon2: Coordinates of the second point
        
    Returns:
        Distance in kilometers
    """
    from math import radians, sin, cos, sqrt, atan2
    
    # Convert to radians
    lat1, lon1 = radians(lat1), radians(lon1)
    lat2, lon2 = radians(lat2), radians(lon2)
    
    # Earth's radius in km
    R = 6371.0
    
    # Differences
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    
    return distance

# Main function to find nearby collection points
def find_collection_points(coordinates: Dict[str, float], waste_type: str, radius: float = 10.0) -> List[Dict[str, Any]]:
    """
    Finds nearby collection points for a specific waste type
    
    Args:
        coordinates: Dictionary containing latitude and longitude of starting point
        waste_type: Type of waste being searched for
        radius: Search radius in kilometers (default 10km)
        
    Returns:
        List of collection points sorted by distance
    """
    if not coordinates:
        return []
    
    # Get all collection points
    all_points = get_collection_points_api(waste_type)
    
    # Calculate distance for each point and filter by radius
    nearby_points = []
    
    for point in all_points:
        distance = calculate_distance(
            coordinates["lat"], coordinates["lon"],
            point["lat"], point["lon"]
        )
        
        # Convert to kilometers and round to 2 decimal places
        point["distance"] = round(distance, 2)
        
        # Filter by radius
        if distance <= radius:
            nearby_points.append(point)
    
    # Sort by distance
    nearby_points.sort(key=lambda x: x["distance"])
    
    return nearby_points

# Function to retrieve collection dates
def get_collection_dates(waste_type: str, address: str) -> List[Dict[str, Any]]:
    """
    Retrieves upcoming collection dates for a waste type and address
    
    Args:
        waste_type: Type of waste (e.g., "Kehricht", "Karton", "Papier", etc.)
        address: Address for which to search for dates
        
    Returns:
        List of upcoming collection dates
    """
    try:
        # Extract street name from address
        address_parts = address.split(',')
        if len(address_parts) < 1:
            return []
        
        street_name = address_parts[0].strip()
        
        # Get current year
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
        
        # Convert waste_type to terms used by the API
        waste_type_mapping = {
            "Aluminium": "Alu+Weissblech",
            "Dosen": "Alu+Weissblech",
            "Glas": "Glas",
            "Papier": "Papier",
            "Karton": "Karton",
            "Kehricht": "Kehricht",
            "Altmetall": "Metall",
            "Sonderabfall": "Sonderabfall",
            "Grüngut": "Grüngut",
            "Alttextilien": "Alttextilien",
            "Altöl": "Altöl",
            "Styropor": "Styropor"
        }
        
        api_waste_type = waste_type_mapping.get(waste_type, waste_type)
        
        # Filter results
        collection_dates = []
        
        for entry in data.get("results", []):
            fields = entry.get("fields", {}) or entry.get("record", {}).get("fields", {})
            
            # Check if collection type matches
            if fields.get("sammlung") == api_waste_type:
                # Check if street is in list of affected streets
                streets = fields.get("strasse", [])
                
                # Handle case where streets might be a string
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
    """
    Formats collection points for display in Streamlit
    
    Args:
        collection_points: List of collection points
        
    Returns:
        Tuple containing (DataFrame for map, Original list of points)
    """
    if not collection_points:
        return pd.DataFrame(columns=["lat", "lon", "name"]), []
    
    # DataFrame for map
    map_data = pd.DataFrame(
        [[point["lat"], point["lon"], point["name"]] for point in collection_points],
        columns=["lat", "lon", "name"]
    )
    
    return map_data, collection_points

# Utility function to translate waste types
def translate_waste_type(waste_type: str) -> str:
    """
    Translates waste types from the API to English
    
    Args:
        waste_type: Waste type in German
        
    Returns:
        Waste type translated to English
    """
    translations = {
        "Aluminium": "Aluminium",
        "Dosen": "Cans",
        "Glas": "Glass",
        "Papier": "Paper",
        "Karton": "Cardboard",
        "Kehricht": "Household waste",
        "Altmetall": "Metal",
        "Sonderabfall": "Hazardous waste",
        "Grüngut": "Green waste",
        "Alttextilien": "Textiles",
        "Altöl": "Oil",
        "Styropor": "Foam packaging"
    }
    
    return translations.get(waste_type, waste_type)

# Function to get the list of all available waste types
def get_available_waste_types() -> List[str]:
    """
    Gets the list of all waste types available in the API
    
    Returns:
        List of waste types
    """
    # Predefined list of waste types
    waste_types = [
        "Kehricht",    # Household waste
        "Papier",      # Paper
        "Karton",      # Cardboard
        "Glas",        # Glass
        "Grüngut",     # Green waste
        "Dosen",       # Cans
        "Aluminium",   # Aluminium
        "Altmetall",   # Metal
        "Alttextilien", # Textiles
        "Altöl",       # Oil
        "Sonderabfall", # Hazardous waste
        "Styropor"     # Foam packaging
    ]
    
    return waste_types