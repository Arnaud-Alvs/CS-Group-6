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
        
        # Ensure address is properly formatted and URL-encoded
        # Add St. Gallen, Switzerland to improve results
        if "st.gallen" not in address.lower() and "st. gallen" not in address.lower():
            address = f"{address}, St. Gallen, Switzerland"
            
        params = {
            "q": address,
            "format": "json",
            "limit": 1,
            "addressdetails": 1
        }
        
        headers = {
            "User-Agent": "WasteWise App - University Project"
        }
        
        # Add a delay to respect Nominatim's usage conditions
        time.sleep(1)
        
        # Debug info
        print(f"Sending request to: {base_url} with params: {params}")
        
        response = requests.get(base_url, params=params, headers=headers)
        
        if response.status_code != 200:
            st.error(f"Geocoding API error: {response.status_code}")
            print(f"API Error: {response.text}")
            return None
            
        data = response.json()
        print(f"API Response: {data}")
        
        if data and len(data) > 0:
            return {
                "lat": float(data[0]["lat"]),
                "lon": float(data[0]["lon"])
            }
        else:
            st.error(f"Unable to find coordinates for address: {address}")
            return None
            
    except Exception as e:
        st.error(f"Error while searching for coordinates: {str(e)}")
        print(f"Exception in get_coordinates: {str(e)}")
        return None

# Fallback collection points for when the API is unavailable
def get_fallback_collection_points(waste_type: str) -> List[Dict[str, Any]]:
    """
    Provides fallback collection points when the API is unavailable
    
    Args:
        waste_type: Type of waste being searched for
        
    Returns:
        List of fallback collection points
    """
    # Common collection points in St. Gallen that accept most waste types
    fallback_points = [
        {
            "id": "fallback_01",
            "name": "St. Gallen Main Recycling Center",
            "address": "Kehrichtverbrennungsanlage KVA, Hüettenwiesstrasse 50, 9014 St. Gallen",
            "lat": 47.4245,
            "lon": 9.3767,
            "waste_types": ["Kehricht", "Papier", "Karton", "Glas", "Altmetall", "Sonderabfall", "Alttextilien", "Altöl", "Styropor"],
            "hours": "Monday-Friday 8:00-17:00, Saturday 8:00-12:00",
            "distance": 0
        },
        {
            "id": "fallback_02",
            "name": "Rosenberg Public Collection Point",
            "address": "Rosenbergstrasse 79, 9000 St. Gallen",
            "lat": 47.4209,
            "lon": 9.3696,
            "waste_types": ["Papier", "Karton", "Glas", "Dosen", "Aluminium"],
            "hours": "24/7 accessible",
            "distance": 0
        },
        {
            "id": "fallback_03",
            "name": "Lachen Collection Center",
            "address": "Lachenstrasse 15, 9013 St. Gallen",
            "lat": 47.4283,
            "lon": 9.3821,
            "waste_types": ["Kehricht", "Grüngut", "Papier", "Karton", "Glas"],
            "hours": "Daily 7:00-20:00",
            "distance": 0
        }
    ]
    
    # Filter points based on waste type
    filtered_points = []
    for point in fallback_points:
        if waste_type in point["waste_types"]:
            filtered_points.append(point)
    
    return filtered_points

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
            "limit": 100,
            "timeout": 30  # Add timeout parameter
        }
        
        # Set timeout for the request
        response = requests.get(api_url, params=params, timeout=30)
        
        if response.status_code != 200:
            st.error(f"Collection points API error: {response.status_code}")
            print(f"API Error: {response.text}")
            return get_fallback_collection_points(waste_type)
            
        data = response.json()
        print("API Response sample:", json.dumps(data.get("results", [])[0] if data.get("results") else {}, indent=2))
        
        collection_points = []
        
        for point in data.get("results", []):
            # Extract the fields (need to check the exact field structure)
            fields = point.get("record", {}).get("fields", {})
            
            # Check if the waste type is accepted at this collection point
            # Try different possible field names for waste types
            waste_types = fields.get("abfallarten", [])
            if not waste_types:
                waste_types = fields.get("waste_types", [])
            if not waste_types:
                waste_types = fields.get("accepted_waste", [])
            
            # If waste_types is a string, split it into a list
            if isinstance(waste_types, str):
                waste_types = [w.strip() for w in waste_types.split(',')]
            
            # Handle case where waste type might be in different format
            if waste_type in waste_types or any(waste_type.lower() in wt.lower() for wt in waste_types):
                # Get coordinates - try different possible structures
                lat = fields.get("lat", None)
                lon = fields.get("lon", None)
                
                # Try alternative structure for coordinates
                if not lat or not lon:
                    geo = point.get("geometry", {})
                    coordinates = geo.get("coordinates", [])
                    if len(coordinates) >= 2:
                        lon, lat = coordinates  # GeoJSON format: [longitude, latitude]
                
                # Try another structure
                if not lat or not lon:
                    geo = fields.get("geo_point_2d", {})
                    if isinstance(geo, dict):
                        lat = geo.get("lat", 0)
                        lon = geo.get("lon", 0)
                    elif isinstance(geo, list) and len(geo) >= 2:
                        lat, lon = geo  # [latitude, longitude]
                
                # Get other information
                sammelstel = fields.get("sammelstel", "")
                standort = fields.get("standort", "Address not specified")
                oeffnungsz = fields.get("oeffnungsz", "Hours not specified")
                
                if lat and lon:
                    collection_point = {
                        "id": sammelstel,
                        "name": f"Collection Point {sammelstel}",
                        "address": standort,
                        "lat": float(lat),
                        "lon": float(lon),
                        "waste_types": waste_types,
                        "hours": oeffnungsz,
                        "distance": 0  # Will be calculated later
                    }
                    collection_points.append(collection_point)
        
        print(f"Found {len(collection_points)} collection points for {waste_type}")
        return collection_points
        
    except requests.exceptions.RequestException as e:
        # Handle network-related errors specifically
        st.error(f"Network error while contacting the API: Unable to connect to daten.stadt.sg.ch")
        st.info("This might be due to network connectivity issues or the API being temporarily unavailable.")
        return get_fallback_collection_points(waste_type)
        
    except Exception as e:
        st.error(f"Error retrieving collection points: {str(e)}")
        print(f"Exception: {str(e)}")
        print(f"Type of error: {type(e)}")
        return get_fallback_collection_points(waste_type)

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
def find_collection_points(coordinates: Dict[str, float], waste_type: str, radius: float = 5.0) -> List[Dict[str, Any]]:
    """
    Finds nearby collection points for a specific waste type
    
    Args:
        coordinates: Dictionary containing latitude and longitude of starting point
        waste_type: Type of waste being searched for
        radius: Search radius in kilometers
        
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
    
    print(f"Found {len(nearby_points)} collection points within {radius} km")
    return nearby_points

# Fallback collection dates for when the API is unavailable
def get_fallback_collection_dates(waste_type: str) -> List[Dict[str, Any]]:
    """
    Provides fallback collection dates when the API is unavailable
    
    Args:
        waste_type: Type of waste
        
    Returns:
        List of fallback collection dates
    """
    # Generate some example collection dates for the current month
    today = datetime.now()
    current_month = today.month
    current_year = today.year
    
    # Common collection schedules for different waste types
    collection_schedules = {
        "Kehricht": [7, 14, 21, 28],  # Monthly on these days
        "Papier": [10, 24],  # Twice per month
        "Karton": [10, 24],  # Same as paper
        "Grüngut": [3, 17, 31],  # Three times per month
        "Sonderabfall": [15]  # Once per month
    }
    
    collection_dates = []
    days = collection_schedules.get(waste_type, [15])  # Default to 15th of month
    
    for day in days:
        if day <= 31:  # Ensure valid day
            try:
                collection_date = datetime(current_year, current_month, day)
                if collection_date >= today:
                    collection_dates.append({
                        "date": collection_date.strftime("%Y-%m-%d"),
                        "time": "07:00",
                        "area": "St. Gallen Center",
                        "title": f"{translate_waste_type(waste_type)} Collection",
                        "pdf": None
                    })
            except ValueError:
                continue
    
    return collection_dates

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
        # Expected format: "Street, Number, City"
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
            "refine.datum": str(current_year)  # Filter by current year
        }
        
        response = requests.get(api_url, params=params, timeout=30)
        
        if response.status_code != 200:
            st.error(f"Collection dates API error: {response.status_code}")
            return get_fallback_collection_dates(waste_type)
            
        data = response.json()
        
        # Convert waste_type to terms used by the API
        waste_type_mapping = {
            "Aluminium": "Alu+Weissblech",
            "Dosen": "Alu+Weissblech",
            "Glas": "Glas",
            "Papier": "Papier",
            "Karton": "Karton",
            "Kehricht": "Kehricht",  # Household waste
            "Altmetall": "Metall",  # Metal
            "Sonderabfall": "Sonderabfall",  # Hazardous waste
            "Grüngut": "Grüngut",  # Green waste
            "Alttextilien": "Alttextilien",  # Textiles
            "Altöl": "Altöl",  # Oil
            "Styropor": "Styropor"  # Foam packaging
        }
        
        api_waste_type = waste_type_mapping.get(waste_type, waste_type)
        
        # Filter results
        collection_dates = []
        
        for entry in data.get("results", []):
            # Extract fields from the record
            fields = entry.get("record", {}).get("fields", {})
            
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
        
    except requests.exceptions.RequestException as e:
        # Handle network-related errors specifically
        st.error(f"Network error while contacting the API: Unable to connect to daten.stadt.sg.ch")
        st.info("This might be due to network connectivity issues or the API being temporarily unavailable.")
        return get_fallback_collection_dates(waste_type)
        
    except Exception as e:
        st.error(f"Error retrieving collection dates: {str(e)}")
        return get_fallback_collection_dates(waste_type)

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