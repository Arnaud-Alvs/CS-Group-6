import requests
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from math import radians, sin, cos, sqrt, atan2
import re # Import regex module
import folium
from streamlit_folium import folium_static

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
    Strictly enforces results within St. Gallen city boundaries.
    """
    try:
        import time
        # Add a small delay to avoid hitting rate limits
        time.sleep(1)
        
        # Define St. Gallen city boundaries (bounding box)
        ST_GALLEN_BOUNDS = {
            "min_lat": 47.3600,  # South boundary
            "max_lat": 47.4800,  # North boundary
            "min_lon": 9.3000,   # West boundary
            "max_lon": 9.4500    # East boundary
        }
        
        base_url = "https://nominatim.openstreetmap.org/search"

        # Add "St. Gallen" to the address if not already present
        if "st. gallen" not in address.lower() and "st.gallen" not in address.lower().replace(" ", ""):
            search_address = f"{address}, St. Gallen, Switzerland"
        else:
            search_address = address
            
        # Use q parameter with full address including city
        params = {
            "q": search_address,
            "format": "json",
            "limit": 5,  # Get multiple results to find one in St. Gallen
            "addressdetails": 1
        }

        # Nominatim REQUIRES a unique user agent identifying your application
        headers = {
            "User-Agent": "WasteWise-StGallen-App/1.0 (university.project@example.com)",
            "Accept-Language": "de,en"
        }

        logger.info(f"Searching for coordinates for: '{search_address}'")
        response = requests.get(base_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        
        # If no results found
        if not data:
            st.warning(f"Could not find coordinates for address: {address}. Please try a more specific address.")
            logger.warning(f"No results found for address: {search_address}")
            return None
            
        # Check each result to find one within St. Gallen boundaries
        for result in data:
            try:
                lat = float(result["lat"])
                lon = float(result["lon"])
                
                # Check if coordinates are within St. Gallen boundaries
                if (ST_GALLEN_BOUNDS["min_lat"] <= lat <= ST_GALLEN_BOUNDS["max_lat"] and 
                    ST_GALLEN_BOUNDS["min_lon"] <= lon <= ST_GALLEN_BOUNDS["max_lon"]):
                    
                    # Debug log to verify the coordinates look correct
                    logger.info(f"Found address in St. Gallen bounds: {result.get('display_name')} at {lat}, {lon}")
                    
                    # Double-check if address contains any St. Gallen reference
                    display_name = result.get('display_name', '').lower()
                    address_parts = result.get('address', {})
                    
                    # Check if any address part mentions St. Gallen
                    st_gallen_mentioned = any("gallen" in str(value).lower() 
                                             for key, value in address_parts.items())
                    
                    if "gallen" in display_name or st_gallen_mentioned:
                        return {"lat": lat, "lon": lon}
            except (ValueError, KeyError) as e:
                logger.warning(f"Error processing result: {e}")
                continue
                
        # If we got here, we didn't find a suitable location in St. Gallen
        st.warning("The address was found, but appears to be outside St. Gallen city. This app is for St. Gallen addresses only.")
        logger.warning(f"No results within St. Gallen boundaries for: {search_address}")
        return None

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching coordinates for {address}: {str(e)}. Please check the address format.")
        logger.error(f"Request error for address {address}: {str(e)}")
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
    
    # For debugging the distance issue
    logger.info(f"Calculated distance: {distance} km between ({lat1}, {lon1}) and ({lat2}, {lon2})")
    
    return distance

# Add a new function to create an interactive map
def create_interactive_map(user_coords: Dict[str, float], collection_points: List[Dict[str, Any]]) -> folium.Map:
    """
    Creates an interactive Folium map with the user's location and nearby collection points.
    Each point shows information on hover.
    """
    # Create a map centered on the user's location
    st_gallen_center = [47.4245, 9.3767]  # Center of St. Gallen
    
    # If we have user coordinates, center on them, otherwise use St. Gallen center
    if user_coords:
        center = [user_coords["lat"], user_coords["lon"]]
    else:
        center = st_gallen_center
    
    # Create the base map
    m = folium.Map(
        location=center,
        zoom_start=14,
        tiles="CartoDB positron",  # A cleaner, more modern map style
        max_bounds=True,  # Restrict panning to the bounds
    )
    
    # Set bounds to restrict to St. Gallen area
    # These are approximate coordinates that define the St. Gallen city area
    sw = [47.3745, 9.3167]  # Southwest corner
    ne = [47.4745, 9.4367]  # Northeast corner
    m.fit_bounds([sw, ne])
    
    # Add user marker
    if user_coords:
        folium.Marker(
            location=[user_coords["lat"], user_coords["lon"]],
            popup="Your Location",
            tooltip="Your Location",
            icon=folium.Icon(color="blue", icon="home", prefix="fa")
        ).add_to(m)
    
    # Add collection points markers
    for point in collection_points:
        # Create a nice tooltip with waste types
        waste_types_str = ", ".join([translate_waste_type(wt) for wt in point["waste_types"]])
        tooltip = f"{point['name']}<br>Accepts: {waste_types_str}"
        
        # Create a popup with more detailed information
        popup_html = f"""
        <div style="width: 200px">
            <h4>{point['name']}</h4>
            <p><b>Distance:</b> {point['distance']:.2f} km</p>
            <p><b>Accepts:</b> {waste_types_str}</p>
            {f"<p><b>Opening Hours:</b> {point['opening_hours']}</p>" if point['opening_hours'] and point['opening_hours'] != "N/A" else ""}
        </div>
        """
        
        # Create the marker with custom icon and add to map
        folium.Marker(
            location=[point["lat"], point["lon"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=tooltip,
            icon=folium.Icon(color="green", icon="recycle", prefix="fa")
        ).add_to(m)
    
    return m

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
    Uses a very simple request to avoid Bad Request errors.
    """
    try:
        # Define the base endpoint without any parameters
        base_endpoint = f"{BASE_API_URL}/api/explore/v2.1/catalog/datasets/abfuhrdaten-stadt-stgallen/records"
        
        logger.info(f"Fetching collection dates from base endpoint: {base_endpoint}")
        
        # Make a simple request without additional parameters
        response = requests.get(base_endpoint, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        results = data.get('results', [])
        
        if results:
            logger.info(f"Successfully fetched {len(results)} collection dates.")
            
            # Filter the results to include only what we need
            filtered_results = []
            for item in results:
                if "sammlung" in item and "strasse" in item and "datum" in item:
                    # Include only the essential fields to reduce data size
                    filtered_item = {
                        "sammlung": item["sammlung"],
                        "strasse": item["strasse"],
                        "datum": item["datum"],
                        "zeit": item.get("zeit", "N/A"),
                        "titel": item.get("titel", "Collection"),
                        "gebietsbezeichnung": item.get("gebietsbezeichnung", "N/A")
                    }
                    filtered_results.append(filtered_item)
            
            logger.info(f"Filtered to {len(filtered_results)} essential collection records.")
            return filtered_results
        else:
            logger.warning("API returned empty results despite successful connection.")
            st.warning("No collection dates found in the database. This may be a temporary issue.")
            return []
            
    except requests.exceptions.RequestException as e:
        error_message = str(e)
        logger.error(f"Error fetching collection dates: {error_message}")
        
        st.error("Unable to connect to the collection dates database. Please try again later.")
        
        # For development/testing, we can use a small sample dataset
        sample_data = [
            {
                "sammlung": "Kehricht",
                "strasse": ["Heimatstrasse", "Bahnhofstrasse"],
                "datum": "2025-05-20",
                "zeit": "ab 7.00 Uhr",
                "titel": "Kehrichtsammlung",
                "gebietsbezeichnung": "Zentrum"
            },
            {
                "sammlung": "Papier",
                "strasse": ["Heimatstrasse", "Bahnhofstrasse"],
                "datum": "2025-05-15",
                "zeit": "ab 7.00 Uhr",
                "titel": "Papiersammlung",
                "gebietsbezeichnung": "Zentrum"
            },
            {
                "sammlung": "Karton",
                "strasse": ["Heimatstrasse", "Bahnhofstrasse"],
                "datum": "2025-06-04",
                "zeit": "ab 7.00 Uhr",
                "titel": "Kartonsammlung",
                "gebietsbezeichnung": "Zentrum"
            },
            {
                "sammlung": "Aluminium",
                "strasse": ["Heimatstrasse", "Bahnhofstrasse"],
                "datum": "2025-06-10",
                "zeit": "ab 7.00 Uhr",
                "titel": "Aluminiumsammlung",
                "gebietsbezeichnung": "Zentrum"
            }
        ]
        
        # In development mode, return the sample data
        logger.info("Using sample data for testing")
        return sample_data
        
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
    Improved with better street name matching.
    """
    today = datetime.now().date()
    relevant_dates = []
    
    # Clean and simplify the street name for better matching
    cleaned_street_name = street_name.lower()
    # Remove street suffixes and common words
    cleaned_street_name = re.sub(r'strasse$|straße$|weg$|gasse$|straat$|allee$', '', cleaned_street_name).strip()
    
    logger.info(f"Searching for collection dates for '{waste_type}' on street '{street_name}' (cleaned: '{cleaned_street_name}')")
    
    # Keep a simple counter to verify we're processing data
    processed_items = 0
    
    for item in all_dates:
        processed_items += 1
        
        # Skip items that don't have the required fields
        if "sammlung" not in item or "strasse" not in item or "datum" not in item:
            continue
            
        # Check if this item is for the waste type we're looking for
        if item["sammlung"].lower() != waste_type.lower():
            continue
            
        # Get the list of streets for this collection
        streets_data = item.get("strasse", [])
        
        # Convert to list if it's not already
        if not isinstance(streets_data, list):
            streets_data = [streets_data]
        
        # Create a simple flag for debugging
        found_match = False
        
        # Check each street in the list
        for street in streets_data:
            if not isinstance(street, str):
                continue
                
            # Clean the street name from database for better matching
            db_street = street.lower()
            db_street = re.sub(r'strasse$|straße$|weg$|gasse$|straat$|allee$', '', db_street).strip()
            
            # Try very simple substring matching
            if cleaned_street_name in db_street or db_street in cleaned_street_name:
                found_match = True
                break
        
        if found_match:
            try:
                # Parse the date
                collection_date_str = item.get("datum")
                if collection_date_str:
                    collection_date = datetime.strptime(collection_date_str, "%Y-%m-%d").date()
                    if collection_date >= today:
                        relevant_dates.append({
                            "date": collection_date,
                            "time": item.get("zeit", "N/A"),
                            "description": item.get("titel", "Collection"),
                            "area": item.get("gebietsbezeichnung", "N/A")
                        })
                        logger.info(f"Added relevant date: {collection_date} for {waste_type}")
            except ValueError:
                logger.warning(f"Invalid date format in collection data: {item['datum']}")
                continue
    
    # Log how many items we processed
    logger.info(f"Processed {processed_items} collection date records")
    
    # Sort relevant dates to find the soonest
    relevant_dates.sort(key=lambda x: x["date"])
    
    if relevant_dates:
        logger.info(f"Found {len(relevant_dates)} future collection dates. Next is on {relevant_dates[0]['date']}")
    else:
        logger.warning(f"No future collection dates found for waste type '{waste_type}' on street '{street_name}'")
        
        # Debug: Log the first few streets from collection dates for this waste type
        streets_sample = []
        for item in all_dates[:20]:  # Look at first 20 items
            if item.get("sammlung", "").lower() == waste_type.lower():
                streets = item.get("strasse", [])
                if isinstance(streets, list) and streets:
                    streets_sample.extend(streets[:3])  # Add up to 3 streets from each matching record
                    
        if streets_sample:
            logger.info(f"Sample streets for {waste_type}: {streets_sample[:10]}")
    
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
    return ["Kehricht", "Papier", "Karton", "Grüngut", "Altmetall", "Glas", "Dosen","Aluminium", "Alttextilien", "Altöl", "Sonderabfall", "Styropor"]

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
    street_parts = address.split()
    if len(street_parts) >= 2:
    # Extract full street name without number
        street_number_pattern = r'\d+'
        street_parts_without_numbers = [part for part in street_parts if not re.match(street_number_pattern, part)]
        street_name = ' '.join(street_parts_without_numbers)
    else:
        street_name = address

    logger.info(f"Extracted street name '{street_name}' from address '{address}'")    # Extract street name from address
    next_date = get_next_collection_date(street_name, waste_type, all_dates)
    
    results["next_collection_date"] = next_date
    results["has_scheduled_collection"] = next_date is not None
    
    # 3. Generate appropriate message based on results
    # Get the translated waste type for user-friendly messages
    waste_type_display = translate_waste_type(waste_type)
    
    if results["has_disposal_locations"] and results["has_scheduled_collection"]:
        # Both options are available - this is the key scenario you highlighted
        collection_date_str = next_date['date'].strftime('%A, %B %d, %Y')
        collection_time_str = next_date.get('time', '')
        
        results["message"] = (
            f"You have two options for {waste_type_display}:\n\n"
            f"1. **Collection from home**: The next collection is on {collection_date_str} {collection_time_str}\n\n"
            f"2. **Drop-off locations**: There are {len(collection_points)} disposal points nearby (see map below)"
        )
    elif results["has_disposal_locations"]:
        results["message"] = (
            f"{waste_type_display} can be dropped off at {len(collection_points)} nearby locations. "
            f"There is no scheduled home collection service for this waste type in your area."
        )
    elif results["has_scheduled_collection"]:
        collection_date_str = next_date['date'].strftime('%A, %B %d, %Y')
        collection_time_str = next_date.get('time', '')
        
        results["message"] = (
            f"{waste_type_display} will be collected on {collection_date_str} {collection_time_str}. "
            f"This waste type is typically collected directly from homes in your area."
        )
    else:
        # Try to check if this waste type is typically collected
        typical_collection_types = ["Papier", "Karton", "Kehricht", "Grüngut"]
        if waste_type in typical_collection_types:
            results["message"] = (
                f"No upcoming collection dates found for {waste_type_display} at your address. "
                f"This waste type is typically collected from homes. Please check the official schedule "
                f"or contact the local waste management office for more information."
            )
        else:
            results["message"] = (
                f"No disposal options found for {waste_type_display}. "
                f"Please check the waste type or contact the local waste management office."
            )
    
    return results

# Note: The original code included endpoints as constants at the top.
# We will keep them there for consistency.
