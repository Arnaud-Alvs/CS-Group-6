import requests
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from math import radians, sin, cos, sqrt, atan2
import re # Import regex module

# Configure logging for this module
import logging
logger = logging.getLogger(__name__)
# Prevent duplicate handlers if Streamlit re-runs the script
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# Base API URL
BASE_API_URL = "https://daten.stadt.sg.ch"

# API Endpoints
COLLECTION_POINTS_ENDPOINT = f"{BASE_API_URL}/api/explore/v2.1/catalog/datasets/sammelstellen/records"
COLLECTION_DATES_ENDPOINT = f"{BASE_API_URL}/api/explore/v2.1/catalog/datasets/abfuhrdaten-stadt-stgallen/records"

# Function to get geographic coordinates (latitude, longitude) from an address
def get_coordinates(address: str, api_key: Optional[str] = None) -> Optional[Dict[str, float]]:
    """
    Get latitude and longitude from address using OpenStreetMap Nominatim API.
    """
    try:
        import time
        # Add a small delay to avoid hitting rate limits
        time.sleep(1)  
        
        base_url = "https://nominatim.openstreetmap.org/search"

        # --- Address Cleaning ---
        # Clean the address as you were doing before
        cleaned_address = address
        cleaned_address = re.sub(r'\b\d{4}\b', '', cleaned_address, flags=re.IGNORECASE).strip()
        
        patterns_to_remove_city_country = [
            r'\bSt\.\s*Gallen\b',
            r'\bStGallen\b',
            r'\bSt\. Gallen\b',
            r'\bSwitzerland\b',
            r'\bSchweiz\b',
        ]

        for pattern in patterns_to_remove_city_country:
             cleaned_address = re.sub(pattern, '', cleaned_address, flags=re.IGNORECASE).strip()

        cleaned_address = re.sub(r'^\s*[,;:\s]+|[,\s]+$', '', cleaned_address).strip()
        cleaned_address = re.sub(r'\s+', ' ', cleaned_address).strip()

        address_for_query = cleaned_address

        if not address_for_query:
             st.warning("Please enter a valid street name and number.")
             logger.warning(f"Empty address string after cleaning: {address}")
             return None

        # Create a more specific query for Switzerland/St. Gallen
        structured_query = f"{address_for_query}, St. Gallen, Switzerland"
        
        params = {
            "q": structured_query,  # Use a structured query format
            "format": "json",
            "limit": 1,
            "addressdetails": 1,
        }

        # Nominatim REQUIRES a unique user agent identifying your application
        headers = {
            "User-Agent": "WasteWise-StGallen-App/1.0 (university.project@example.com)",
            "Accept-Language": "de,en"  # Add German as preferred language
        }

        logger.info(f"Attempting to get coordinates for original address: '{address}', query: '{structured_query}'")
        response = requests.get(base_url, params=params, headers=headers, timeout=30)

        # Raise an HTTPError for bad responses (4xx or 5xx)
        response.raise_for_status()

        data = response.json()

        if data and len(data) > 0:
            logger.info(f"Successfully retrieved coordinates for {address}: {data[0]['lat']}, {data[0]['lon']}")
            # Return latitude and longitude as a dictionary
            return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"])}
        else:
            st.warning(f"Could not find coordinates for address: {address}. Please try a more specific address.")
            logger.warning(f"Nominatim found no results for address: {address}, query: {structured_query}")
            return None

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching coordinates for {address}: {str(e)}. Please check the address format.")
        logger.error(f"Request error for address {address}, query {structured_query if 'structured_query' in locals() else 'unknown'}: {str(e)}")
        return None
    except ValueError as e:
        st.error(f"Error parsing coordinate data for {address}: {str(e)}")
        logger.error(f"Value error for address {address}: {str(e)}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while getting coordinates for {address}: {str(e)}")
        logger.error(f"Unexpected error for address {address}: {str(e)}")
        return None

# Function to calculate the distance between two points using the Haversine formula
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on the Earth
    (specified in decimal degrees) using the Haversine formula.
    Returns distance in kilometers.
    """
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert degrees to radians
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    # Difference in coordinates
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# Function to fetch collection points data from the API
def fetch_collection_points() -> List[Dict[str, Any]]:
    """
    Fetches waste collection points data from the St. Gallen Open Data API.
    """
    try:
        # Set a higher limit to get more results, or remove it to use default
        params = {"limit": 100} # Increased limit for potentially more points
        logger.info(f"Fetching collection points from: {COLLECTION_POINTS_ENDPOINT} with params: {params}")
        response = requests.get(COLLECTION_POINTS_ENDPOINT, params=params, timeout=30)
        response.raise_for_status() # Raise an HTTPError for bad responses
        data = response.json()
        logger.info(f"Successfully fetched {len(data.get('results', []))} collection points.")
        # The actual records are in the 'results' key for v2.1
        return data.get('results', [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching collection points data: {str(e)}")
        logger.error(f"Error fetching collection points: {str(e)}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching collection points: {str(e)}")
        logger.error(f"Unexpected error fetching collection points: {str(e)}")
        return []


# Function to fetch collection dates data from the API
def fetch_collection_dates() -> List[Dict[str, Any]]:
    """
    Fetches waste collection dates data from the St. Gallen Open Data API.
    Filters for the current year (or a specified year if needed).
    """
    try:
        # Filter for the current year. Adjust if you need data for other years.
        current_year = datetime.now().year
        # Note: The provided URL snippet uses 2025, so we'll keep that for now.
        # If you need the current year dynamically, replace "2025" with {current_year}
        params = {"limit": 1000, "refine": "datum:\"2025\""} # Increased limit and refined by year
        logger.info(f"Fetching collection dates from: {COLLECTION_DATES_ENDPOINT} with params: {params}")
        response = requests.get(COLLECTION_DATES_ENDPOINT, params=params, timeout=30)
        response.raise_for_status() # Raise an HTTPError for bad responses
        data = response.json()
        logger.info(f"Successfully fetched {len(data.get('results', []))} collection dates.")
        # The actual records are in the 'results' key for v2.1
        return data.get('results', [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching collection dates data: {str(e)}")
        logger.error(f"Error fetching collection dates: {str(e)}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching collection dates: {str(e)}")
        logger.error(f"Unexpected error fetching collection dates: {str(e)}")
        return []

# Function to find nearest collection points for a given waste type and user location
def find_collection_points(user_lat: float, user_lon: float, waste_type: str, all_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Finds collection points that accept the specified waste type and calculates
    their distance from the user's location. Returns a list of points sorted by distance.
    """
    suitable_points = []
    
    # Iterate through all fetched collection points
    for point in all_points:
        # Check if the point has location data and accepts the waste type
        if "geo_point_2d" in point and point["geo_point_2d"] and "abfallarten" in point:
            # Case-insensitive check for the waste type
            waste_types_at_point = [wt.lower() for wt in point.get("abfallarten", [])]
            
            if waste_type.lower() in waste_types_at_point:
                try:
                    point_lat = float(point["geo_point_2d"]["lat"])
                    point_lon = float(point["geo_point_2d"]["lon"])
                    
                    # Calculate distance
                    distance = haversine_distance(user_lat, user_lon, point_lat, point_lon)
                    
                    # Add distance and formatted address to the point data
                    point_data = {
                        "name": point.get("standort", "Unknown Location"),
                        "lat": point_lat,
                        "lon": point_lon,
                        "distance": distance,
                        "waste_types": point.get("abfallarten", []),
                        "opening_hours": point.get("oeffnungsz", "N/A") 
                    }
                    suitable_points.append(point_data)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid collection point data: {str(e)}")
                    continue # Skip this point if data is invalid

    # Sort suitable points by distance
    suitable_points.sort(key=lambda x: x["distance"])
    
    return suitable_points

# Function to get the next collection date for a given waste type and street
def get_next_collection_date(street_name: str, waste_type: str, all_dates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Finds the next collection date for a specific waste type and street.
    """
    today = datetime.now().date()
    relevant_dates = []
    
    # Clean the street name for better matching
    cleaned_street_name = street_name.lower()
    cleaned_street_name = re.sub(r'strasse$|straße$|weg$|gasse$', '', cleaned_street_name).strip()
    
    logger.info(f"Searching for collection dates for '{waste_type}' on street '{street_name}' (cleaned: '{cleaned_street_name}')")
    
    for item in all_dates:
        if "sammlung" not in item or "strasse" not in item or "datum" not in item:
            continue
            
        # Check if this item is for the waste type we're looking for
        if item["sammlung"].lower() != waste_type.lower():
            continue
            
        # Get the list of streets for this collection
        streets = [s.lower() for s in item.get("strasse", [])]
        
        # Check if our street is in the list (using partial matching)
        street_match = False
        for street in streets:
            street_lower = street.lower()
            # Remove common suffixes for better matching
            street_clean = re.sub(r'strasse$|straße$|weg$|gasse$', '', street_lower).strip()
            
            # Check for various matching conditions
            if (cleaned_street_name in street_clean or 
                street_clean in cleaned_street_name or
                street_lower.startswith(cleaned_street_name) or
                cleaned_street_name.startswith(street_clean)):
                street_match = True
                break
                
        if street_match:
            try:
                # Parse the date
                collection_date = datetime.strptime(item["datum"], "%Y-%m-%d").date()
                
                # Only include future dates
                if collection_date >= today:
                    relevant_dates.append({
                        "date": collection_date,
                        "time": item.get("zeit", "N/A"),
                        "description": item.get("titel", "Collection"),
                        "area": item.get("gebietsbezeichnung", "N/A")
                    })
            except ValueError:
                logger.warning(f"Invalid date format in collection data: {item['datum']}")
                continue
    
    # Sort relevant dates to find the soonest
    relevant_dates.sort(key=lambda x: x["date"])
    
    # Return the soonest date, or None if no future dates are found
    return relevant_dates[0] if relevant_dates else None

# Function to format results for display in Streamlit
def format_collection_points(collection_points: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Formats the list of collection points into a pandas DataFrame for map display
    and a list for detailed information display in Streamlit.
    """
    if not collection_points:
        # Return empty DataFrame and list if no points are found
        return pd.DataFrame(columns=["lat", "lon", "name"]), []

    # Create DataFrame for map, including only necessary columns
    map_data = pd.DataFrame([[p["lat"], p["lon"], p["name"]] for p in collection_points], columns=["lat", "lon", "name"])

    # Return the map data and the original list of points (which now includes distance, etc.)
    return map_data, collection_points

# ----------------------------------------
# TRANSLATION (Keep existing translation functions)
# ----------------------------------------

def translate_waste_type(waste_type: str) -> str:
    """
    Translates waste type names from German (from API) to English.
    """
    translations = {
        "Aluminium": "Aluminum",
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
    return translations.get(waste_type, waste_type) # Return original if no translation found

def get_available_waste_types() -> List[str]:
    """
    Returns a list of available waste types based on the collection dates data.
    (This function might need adjustment based on how waste types are listed in sammelstellen.json vs abfuhrdaten-stadt-stgallen.json)
    For now, we'll list common types based on the provided data snippets.
    """
    # Based on the provided JSON snippets, these are the waste types available in the collection dates data.
    # If you need the types from sammelstellen.json, you would need to parse that data source.
    # Let's list the types found in abfuhrdaten-stadt-stgallen.json snippets.
    return ["Kehricht", "Papier", "Karton", "Grüngut", "Altmetall"]

def handle_waste_disposal(address: str, waste_type: str) -> Dict[str, Any]:
    """
    Handles waste disposal lookup based on user address and waste type.
    Returns both collection points and scheduled dates when available.
    """
    results = {
        "waste_type": waste_type,
        "collection_points": [],
        "next_collection_date": None,
        "has_disposal_locations": False,
        "has_scheduled_collection": False,
        "message": ""
    }
    
    # First, get coordinates for the user's address
    coordinates = get_coordinates(address)
    if not coordinates:
        results["message"] = f"Could not find coordinates for address: {address}. Please try a more specific address."
        return results
    
    # Get all collection points
    all_points = fetch_collection_points()
    
    # Get all collection dates
    all_dates = fetch_collection_dates()
    
    # Normalize waste type to match the data sources
    waste_type_original = waste_type
    waste_type_mapping = {
        "paper": "Papier",
        "cardboard": "Karton",
        "household waste": "Kehricht",
        "green waste": "Grüngut",
        "metal": "Altmetall",
        "aluminum": "Aluminium",
        "glass": "Glas",
        "oil": "Altöl",
        "textiles": "Alttextilien",
        "cans": "Dosen"
    }
    
    # Try to normalize the waste type (if in English)
    if waste_type.lower() in waste_type_mapping:
        waste_type = waste_type_mapping[waste_type.lower()]
    
    # 1. Find collection points for the waste type
    collection_points = find_collection_points(
        coordinates["lat"], 
        coordinates["lon"], 
        waste_type, 
        all_points
    )
    
    results["collection_points"] = collection_points
    results["has_disposal_locations"] = len(collection_points) > 0
    
    # 2. Get the next collection date for the waste type
    street_name = address.split()[0]  # Extract street name from address
    next_date = get_next_collection_date(street_name, waste_type, all_dates)
    
    results["next_collection_date"] = next_date
    results["has_scheduled_collection"] = next_date is not None
    
    # 3. Generate appropriate message based on results
    if results["has_disposal_locations"] and results["has_scheduled_collection"]:
        results["message"] = (
            f"{waste_type} can be dropped off at {len(collection_points)} nearby locations "
            f"AND is collected on {next_date['date'].strftime('%A, %B %d, %Y')} {next_date.get('time', '')}"
        )
    elif results["has_disposal_locations"]:
        results["message"] = (
            f"{waste_type} can be dropped off at {len(collection_points)} nearby locations. "
            f"There is no scheduled collection service for this waste type."
        )
    elif results["has_scheduled_collection"]:
        results["message"] = (
            f"{waste_type} will be collected on {next_date['date'].strftime('%A, %B %d, %Y')} {next_date.get('time', '')}. "
            f"There are no drop-off points available for this waste type."
        )
    else:
        results["message"] = (
            f"No disposal options found for {waste_type_original}. "
            f"Please check the waste type or contact the local waste management office."
        )
    
    return results

# Note: The original code included endpoints as constants at the top.
# We will keep them there for consistency.
