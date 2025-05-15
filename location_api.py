# This API module connects our application to the St. Gallen city open data services
# We use it to obtain information about waste collection points and pickup schedules
# The API requires making HTTP requests to retrieve data from the city's endpoints
# We import various libraries to handle data processing, calculations, and visualization

import requests
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from math import radians, sin, cos, sqrt, atan2
import re # Import regex module
import folium
from streamlit_folium import st_folium

# Configure logging for this module to track events and errors
import logging
logger = logging.getLogger(__name__)
# Prevent duplicate handlers if Streamlit re-runs the script (a common issue with Streamlit's execution model)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# Base API URL - This is the root endpoint for all St. Gallen city data services
# Using a constant API helps maintain consistency and makes future URL changes easier
BASE_API_URL = "https://daten.stadt.sg.ch"

# Function to get geographic coordinates (latitude, longitude) from an address
# Collection points are physical locations where waste can be dropped off
# Collection dates are scheduled pickup times for different waste types
COLLECTION_POINTS_ENDPOINT = f"{BASE_API_URL}/api/explore/v2.1/catalog/datasets/sammelstellen/records"
COLLECTION_DATES_ENDPOINT = f"{BASE_API_URL}/api/explore/v2.1/catalog/datasets/abfuhrdaten-stadt-stgallen/records"

# Function to get geographic coordinates (latitude, longitude) from an address
# This is a function that converts user text input into map coordinates
# We use OpenStreetMap's free Nominatim geocoding service with proper attribution
def get_coordinates(address: str, api_key: Optional[str] = None) -> Optional[Dict[str, float]]:
    """
    Get latitude and longitude from address using OpenStreetMap Nominatim API.
    Strictly enforces results within St. Gallen city boundaries.
    """
# Define St. Gallen city boundaries (bounding box)
# This prevents the app from returning results outside our service area
    ST_GALLEN_BOUNDS = {
        "min_lat": 47.3600,  # South boundary
        "max_lat": 47.4800,  # North boundary
        "min_lon": 9.3000,   # West boundary
        "max_lon": 9.4500    # East boundary
    }

    try:
        import time
        # Add a small delay to avoid hitting rate limits
        # Without this delay, our application might be blocked for excessive requests
        time.sleep(1)
        
        
        base_url = "https://nominatim.openstreetmap.org/search"

        # Add "St. Gallen" to the address if not already present
        # This improves geocoding accuracy by providing context to the API

        if "st. gallen" not in address.lower() and "st.gallen" not in address.lower().replace(" ", ""):
            search_address = f"{address}, St. Gallen, Switzerland"
        else:
            search_address = address
            
        # Use q parameter with full address including city
        # These are additional parameters improve result quality and provide address details
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
        
        
        # If no results found, inform the user with a helpful error message
        if not data:
            st.warning(f"Could not find coordinates for address: {address}. Please try a more specific address.")
            logger.warning(f"No results found for address: {search_address}")
            return None
            
        # Check each result to find one within St. Gallen boundaries
        # We might get multiple results, so we need to verify each one
        for result in data:
            try:
                lat = float(result["lat"])
                lon = float(result["lon"])
                
                # Check if coordinates are within St. Gallen boundaries
                # This is our geographic filter to ensure results are relevant
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
        # Provide a clear message to the user about the limitation of our service
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
# This mathematical formula accounts for Earth's curvature when calculating distances
# We use it to determine how far collection points are from the user's location

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on the Earth
    (specified in decimal degrees) using the Haversine formula.
    Returns distance in kilometers.
    """
    # Radius of the Earth in kilometers, a standard constant for these calculations
    R = 6371.0

    # Convert degrees to radians - mathematical operations require radian angles
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    # Difference in coordinates - the core of the Haversine calculation
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    # Haversine formula implementation
    # This accounts for the Earth's spherical shape when calculating distance
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    
    # For debugging the distance issue - helps track calculations in the logs
    logger.info(f"Calculated distance: {distance} km between ({lat1}, {lon1}) and ({lat2}, {lon2})")
    
    return distance

# Add a new function to create an interactive map
# This function visualizes the user's location and nearby collection points
# We use the folium library which creates interactive maps based on Leaflet.js
def create_interactive_map(user_coords: Dict[str, float], collection_points: List[Dict[str, Any]]) -> folium.Map:
    """
    Creates an interactive Folium map with the user's location and nearby collection points.
    Each point shows information on hover.
    """
    # Create a map centered on the user's location
    # If no user location is provided, we default to St. Gallen city center
    st_gallen_center = [47.4245, 9.3767]  # Center of St. Gallen
    
    # If we have user coordinates, center on the user's location, otherwise use St. Gallen center
    if user_coords:
        center = [user_coords["lat"], user_coords["lon"]]
    else:
        center = st_gallen_center
    
    # Create the base map with a cleaner style
    # CartoDB positron gives a clean, light background that makes markers stand out
    m = folium.Map(
        location=center,
        zoom_start=14,
        tiles="CartoDB positron",
        max_bounds=True,  # Enable bounds restriction
    )
  
    # Add a tighter max bounds to prevent users from panning too far
    # This keeps the focus on the St. Gallen area where our data is relevant
    m.options['maxBounds'] = [[47.3600, 9.3000], [47.4800, 9.4500]]

    # Add a watermark/attribution to the map for branding
    # This appears as a small overlay at the bottom left of the map
    m.get_root().html.add_child(folium.Element(
        """
        <div style="position: fixed; bottom: 10px; left: 10px; z-index: 1000; 
                  background-color: white; padding: 5px; border-radius: 5px; font-size: 12px; opacity: 0.8;">
            WasteWise - St. Gallen
        </div>
        """
    ))

    # Add user marker with a nicer icon
    # This shows the user where their entered address is located on the map
    if user_coords:
        folium.Marker(
            location=[user_coords["lat"], user_coords["lon"]],
            popup="Your Location",
            tooltip="Your Location",
            icon=folium.Icon(color="blue", icon="home", prefix="fa")
        ).add_to(m)
    
    # Add collection points markers with improved styling
    # Each waste collection point gets its own marker with detailed information
    for point in collection_points:
        # Create a nice tooltip with waste types
        # This shows basic info when hovering over a point
        waste_types_str = ", ".join([translate_waste_type(wt) for wt in point["waste_types"]])
        tooltip = f"{point['name']}<br>Accepts: {waste_types_str}"
        
        # Create a popup with more detailed information and better styling
        # This appears when clicking on a marker and contains comprehensive information
        popup_html = f"""
        <div style="width: 250px; padding: 10px;">
            <h4 style="color: #2c7fb8; margin-top: 0;">{point['name']}</h4>
            <p><strong>Distance:</strong> {point['distance']:.2f} km</p>
            <p><strong>Accepts:</strong> {waste_types_str}</p>
            {f"<p><strong>Opening Hours:</strong> {point['opening_hours']}</p>" if point['opening_hours'] and point['opening_hours'] != "N/A" else ""}
        </div>
        """
        
        # Create the marker with custom icon and add to map
        # The recycle icon visually indicates the purpose of these locations
        folium.Marker(
            location=[point["lat"], point["lon"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=tooltip,
            icon=folium.Icon(color="green", icon="recycle", prefix="fa")
        ).add_to(m)
    
    return m
    
# Function to fetch collection points data from the API
# This retrieves all waste disposal locations from the city's database
# We use the requests library to make HTTP calls to the city's API
def fetch_collection_points() -> List[Dict[str, Any]]:
    """
    Fetches waste collection points data from the St. Gallen Open Data API.
    """
    try:
        # Set a higher limit to get more results, or remove it to use default
        # The API has pagination, so we request a larger number of results
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
# This retrieves scheduled waste pickup dates from the city's database
# We use pagination to handle potentially large datasets efficiently

def fetch_collection_dates() -> List[Dict[str, Any]]:
    """
    Fetches waste collection dates data from the St. Gallen Open Data API.
    Pre-filters to include only 2025 dates for better performance.
    """
    try:
        all_results = []
        base_url = f"{BASE_API_URL}/api/explore/v2.1/catalog/datasets/abfuhrdaten-stadt-stgallen/records"
        
        # Start with page 0
        offset = 0
        limit = 100  # Number of records per page
        
        logger.info(f"Starting to fetch collection dates from {base_url}")

        # Use a while loop to fetch all pages of data
        # This handles pagination by requesting data in chunks until we have everything
        while True:
            
            # Configure parameters for this page, including the 2025 filter
            # We filter by year to reduce data volume and improve performance
            params = {
                "limit": limit,
                "offset": offset,
                "refine": "datum:\"2025\""  # Filter for 2025 dates only
            }
            
            logger.info(f"Fetching page with offset {offset}, limit {limit}, filtered to 2025")
            
            # Make the request to the API endpoint
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            page_results = data.get('results', [])
            total_count = data.get('total_count', 0)

            # If no results on this page, we've reached the end
            if not page_results:
                logger.info(f"No more results at offset {offset}")
                break
                
            # Add this page's results to our collection
            all_results.extend(page_results)
            logger.info(f"Retrieved {len(page_results)} records, total so far: {len(all_results)}")
            
            # If we've got all the results, or if there are no more results, stop
            # This avoids unnecessary API calls once we have all the data
            if len(all_results) >= total_count or len(page_results) < limit:
                logger.info(f"Finished fetching data, got {len(all_results)} of {total_count} total records")
                break
                
            # Otherwise, move to the next page
            offset += limit
        
        if all_results:
            logger.info(f"Successfully fetched {len(all_results)} collection dates for 2025.")
            return all_results
        else:
            logger.warning("API returned empty results despite successful connection.")
            return []
            
    except Exception as e:
        logger.error(f"Error fetching collection dates: {str(e)}")
        st.error(f"Unable to retrieve collection dates from the API: {str(e)}")
        return []
        
# Function to find nearest collection points for a given waste type and user location
# This filters collection points to only those accepting the specified waste type
# We calculate distances to sort by proximity to the user's location
def find_collection_points(user_lat: float, user_lon: float, waste_type: str, all_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Finds collection points that accept the specified waste type and calculates
    their distance from the user's location. Returns a list of points sorted by distance.
    """
    suitable_points = []
    
    # Iterate through all fetched collection points
    # We need to check each one for relevance to the user's query
    for point in all_points:
        # Check if the point has location data and accepts the waste type
        if "geo_point_2d" in point and point["geo_point_2d"] and "abfallarten" in point:
            # Case-insensitive check for the waste type
            waste_types_at_point = [wt.lower() for wt in point.get("abfallarten", [])]
            
            if waste_type.lower() in waste_types_at_point:
                try:
                    point_lat = float(point["geo_point_2d"]["lat"])
                    point_lon = float(point["geo_point_2d"]["lon"])
                    
                    # Calculate distance from user to this collection point
                    # This lets us sort results by proximity
                    distance = haversine_distance(user_lat, user_lon, point_lat, point_lon)
                    
                    # We compile a clean, consistent data structure for display
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
# This helps users know when their waste will be collected from their location
def get_next_collection_date(street_name: str, waste_type: str, all_dates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Finds the next collection date for a specific waste type and street.
    Skips collections that are earlier on the same day.
    """
    # Get current date and time for same-day checks
    # This ensures we don't show collections that have already passed today
    now = datetime.now()
    today = now.date()
    current_hour = now.hour
    relevant_dates = []
    
    # Clean the input street name for matching
    cleaned_street_name = street_name.lower().strip()
    
    logger.info(f"Searching for collection dates for '{waste_type}' on street '{street_name}'")
    
    for item in all_dates:
        # Check if this item is for the waste type we're looking for
        if item.get('sammlung', '').lower() != waste_type.lower():
            continue
            
        # Get the streets list
        streets = item.get('strasse', [])
        
        # Ensure streets is formatted as a list for consistent processing
        if not isinstance(streets, list):
            streets = [streets]
        
        # Check each street for a match
        for street in streets:
            if not isinstance(street, str):
                continue
                
            street_lower = street.lower().strip()
            
            # Check if either is a substring of the other
            if cleaned_street_name in street_lower or street_lower in cleaned_street_name:
                try:
                    # Parse the date
                    date_str = item.get('datum')
                    if date_str:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                        
                        # Check if it's in the future
                        if date_obj > today:
                            # Future date is definitely valid
                            relevant_dates.append({
                                'date': date_obj,
                                'time': item.get('zeit', 'N/A'),
                                'description': item.get('titel', 'Collection'),
                                'area': item.get('gebietsbezeichnung', 'N/A')
                            })
                        elif date_obj == today:
                            # It's today, check the time to see if it's already passed
                            time_str = item.get('zeit', '')
                            
                            # Try to extract the hour from time string like "ab 7.00 Uhr"
                            collection_hour = None
                            time_match = re.search(r'ab\s+(\d+)[\.:]', time_str)
                            if time_match:
                                try:
                                    collection_hour = int(time_match.group(1))
                                    
                                    # If collection time is in the future today, include it
                                    # Skip collections that have already happened today
                                    if collection_hour > current_hour:
                                        # If we can't parse the hour, include it to be safe
                                        # Better to show potentially passed collections than miss future ones
                                        relevant_dates.append({
                                            'date': date_obj,
                                            'time': time_str,
                                            'description': item.get('titel', 'Collection'),
                                            'area': item.get('gebietsbezeichnung', 'N/A')
                                        })
                                    else:
                                        logger.info(f"Skipping same-day collection at {collection_hour}:00 because current time is {current_hour}:00")
                                except ValueError:
                                    # If we can't parse the hour, include it to be safe
                                    relevant_dates.append({
                                        'date': date_obj,
                                        'time': time_str,
                                        'description': item.get('titel', 'Collection'),
                                        'area': item.get('gebietsbezeichnung', 'N/A')
                                    })
                            else:
                                # If we can't extract the hour, include it to be safe
                                relevant_dates.append({
                                    'date': date_obj,
                                    'time': time_str,
                                    'description': item.get('titel', 'Collection'),
                                    'area': item.get('gebietsbezeichnung', 'N/A')
                                })
                except ValueError:
                    logger.warning(f"Invalid date format in collection data: {item.get('datum')}")
                    continue
                
                # Once we find a match for this record, move on to the next one
                break
    
    # Sort relevant dates to find the soonest
    relevant_dates.sort(key=lambda x: x['date'])
    
    if relevant_dates:
        return relevant_dates[0]
    else:
        logger.warning(f"No future collection dates found for waste type '{waste_type}' on street '{street_name}'")
        return None
        
# Function to format results for display in Streamlit
# This prepares data structures that work well with Streamlit's display components
# We create both map data and detailed information for the UI
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

# This function translates waste type names from German to English
# Our API returns data in German, but we want to display information in English
# A simple dictionary lookup handles the common waste type translations
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

# Function to get a list of all supported waste types in our application
# This helps populate dropdowns and validate user input
# The list is based on what's available in the St. Gallen waste system
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

# The main function that handles waste disposal lookup requests
# This is the core functionality that combines address lookup, waste type,
# collection points and scheduled pickups to give users complete information
def handle_waste_disposal(address: str, waste_type: str) -> Dict[str, Any]:
    """
    Handles waste disposal lookup based on user address and waste type.
    Improved to better handle cases where a street exists but doesn't have specific waste collections.
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
    
    # Get all collection points from API
    all_points = fetch_collection_points()
    
    # Get all collection dates from API
    all_dates = fetch_collection_dates()
    
   # Normalize waste type to match the data sources
   # This allows users to enter waste types in English, while our API uses German
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
    
    # Find collection points for the waste type
    collection_points = find_collection_points(
        coordinates["lat"], 
        coordinates["lon"], 
        waste_type, 
        all_points
    )
    
    results["collection_points"] = collection_points
    results["has_disposal_locations"] = len(collection_points) > 0
    
    # Get the next collection date for the waste type
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
    
    # Check for available waste types at this street (for better messaging)
    available_waste_types = set()
    for item in all_dates:
        streets = item.get('strasse', [])
        # Make sure we are always working with a list, even if it's just one street
        if not isinstance(streets, list):
            streets = [streets]
            
        street_found = False
        for s in streets:
            if not isinstance(s, str):
                continue
            # We compare street names    
            if s.lower().strip() == street_name.lower().strip():
                street_found = True
                break # We found a match, no need to check the rest
        # If this street has a collection and a waste type is defined, we store it        
        if street_found and 'sammlung' in item:
            available_waste_types.add(item['sammlung'])
    
    # Generate appropriate message based on results
    # Get the translated waste type for user-friendly messages
    waste_type_display = translate_waste_type(waste_type)

    # CASE 1: Both options available: home collection + nearby drop-off locations
    if results["has_disposal_locations"] and results["has_scheduled_collection"]:
        
        collection_date_str = next_date['date'].strftime('%A, %B %d, %Y') # Format the date nicely
        collection_time_str = next_date.get('time', '') # Get collection time if available
        
        results["message"] = (
            f"You have two options for {waste_type_display}:\n\n"
            f"1. **Collection from home**: The next collection is on {collection_date_str} {collection_time_str}\n\n"
            f"2. **Drop-off locations**: There are {len(collection_points)} disposal points nearby (see map below)"
        )
     # CASE 2: Only drop-off points are available, no home collection    
    elif results["has_disposal_locations"]:
        # Create a more informative message about why there's no collection service
        if available_waste_types:
            # Show the user which types are collected in their area
            available_types_str = ", ".join([translate_waste_type(wt) for wt in available_waste_types])
            results["message"] = (
                f"{waste_type_display} can be dropped off at {len(collection_points)} nearby locations. "
                f"There is no scheduled home collection service for {waste_type_display} on {street_name}. "
                f"Available collection services for your street include: {available_types_str}."
            )
        else:
            results["message"] = (
                f"{waste_type_display} can be dropped off at {len(collection_points)} nearby locations. "
                f"There is no scheduled home collection service for this waste type in your area."
            )
    # CASE 3: Only home collection is available (no drop-off points found)        
    elif results["has_scheduled_collection"]:
        collection_date_str = next_date['date'].strftime('%A, %B %d, %Y')
        collection_time_str = next_date.get('time', '')

    # CASE 4: Neither drop-off points nor home collection found
    else:
        # Try to check if this waste type is typically collected
        # Check if the waste type is usually collected by default (like paper or household waste)
        typical_collection_types = ["Papier", "Karton", "Kehricht", "Grüngut"]
        
        if available_waste_types:
            available_types_str = ", ".join([translate_waste_type(wt) for wt in available_waste_types])
            results["message"] = (
                f"No disposal options found for {waste_type_display} at {street_name}. "
                f"Available collection services for your street include: {available_types_str}. "
                f"You may need to visit a recycling center for {waste_type_display}."
            )
        elif waste_type in typical_collection_types:
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
    # We return the final result, which includes the message, coordinates, date, etc.
    return results

