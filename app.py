import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests
import json
import pickle
from datetime import datetime
import os
import sys
import logging

# Configure error handling and logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import functions from location_api.py
try:
    from location_api import (
        get_coordinates,
        find_collection_points,
        fetch_collection_dates,
        get_next_collection_date,
        format_collection_points,
        get_available_waste_types,
        translate_waste_type,
        fetch_collection_points,
        COLLECTION_POINTS_ENDPOINT,
        COLLECTION_DATES_ENDPOINT,
        handle_waste_disposal,
        create_interactive_map
    )
    logger.info(f"Successfully imported location_api functions")
    logger.info(f"Collection Points API: {COLLECTION_POINTS_ENDPOINT}")
    logger.info(f"Collection Dates API: {COLLECTION_DATES_ENDPOINT}")
except ImportError as e:
    st.error(f"Failed to import from location_api.py: {str(e)}")
    logger.error(f"Import error: {str(e)}")
    st.stop() # Stop execution if the core dependency is missing
except Exception as e:
    st.error(f"An unexpected error occurred while importing location_api.py: {str(e)}")
    logger.error(f"Unexpected error: {str(e)}")
    st.stop() # Stop execution on other import errors

# Initialize session state if not already done
if 'waste_info_results' not in st.session_state:
    st.session_state.waste_info_results = None
    st.session_state.show_results = False

# Function to check if ML models are available (moved to app.py to be accessible by all pages)
def check_ml_models_available():
    """Check if ML model files exist"""
    required_files = ['waste_classifier.pkl', 'waste_vectorizer.pkl', 'waste_encoder.pkl']
    
    for file in required_files:
        if not os.path.exists(file):
            return False
    return True

# Function to check if TensorFlow is available
def check_tensorflow_available():
    """Check if TensorFlow is available"""
    try:
        import tensorflow
        return True
    except ImportError:
        return False

        # Function to load the text model - fixed version
@st.cache_resource
def load_text_model():
    """Load text classification model with proper error handling"""
    try:
        if not check_ml_models_available():
            logger.warning("ML model files not found")
            return None, None, None
            
        with open('waste_classifier.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('waste_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('waste_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        return model, vectorizer, encoder
    except Exception as e:
        logger.error(f"Error loading text model: {str(e)}")
        return None, None, None
# Function to load the text model - fixed version
def download_model_from_reliable_source(model_path):
    """Download model from a reliable source"""
    try:
        if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
            logger.info(f"Model file already exists at {model_path}")
            return True
            
        # URL to your hosted model file
        model_url = "https://github.com/Arnaud-Alvs/CS-Group-6/releases/download/V-1.0.0/waste_image_classifier.h5"
        
        st.info("Downloading image classification model... This may take a moment.")
        
        response = requests.get(model_url, stream=True)
        if response.status_code != 200:
            logger.error(f"Failed to download model: HTTP {response.status_code}")
            return False
            
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"Starting download, expected size: {total_size} bytes")       
        
        if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
            logger.info(f"Model downloaded successfully: {os.path.getsize(model_path)} bytes")
            return True
        else:
            logger.error(f"Download failed or file too small: {os.path.getsize(model_path)} bytes")
            return False
            
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return False


# In app.py, modify your load_image_model function to integrate the new approach
@st.cache_resource
def load_image_model():
    """Load image classification model - focused on reliability"""
    try:
        if not check_tensorflow_available():
            logger.warning("TensorFlow not available")
            return None
            
        import tensorflow as tf
        import os
        
        # Define model path
        model_path = os.path.join(os.path.dirname(__file__), "waste_image_classifier.h5")
        
        # Download the model if needed
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
            success = download_model_from_reliable_source(model_path)
            if not success:
                logger.error("Failed to download model")
                return None
        
        # Load the model with error handling
        try:
            logger.info(f"Loading model from {model_path}")
            model = tf.keras.models.load_model(model_path, compile=False)
            logger.info("Model loaded successfully!")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"Unexpected error in load_image_model: {str(e)}")
        return None

# Enhanced predict_from_text function with fallback
def predict_from_text(description, model=None, vectorizer=None, encoder=None):
    """Predict waste type from text with fallback to rule-based"""
    if not description:
        return None, 0.0
    
    # Use ML model if available
    if model is not None and vectorizer is not None and encoder is not None:
        try:
            description = description.lower()
            X_new = vectorizer.transform([description])
            prediction = model.predict(X_new)[0]
            probabilities = model.predict_proba(X_new)[0]
            confidence = probabilities[prediction]
            
            # Get category from encoder and ensure it's in the right format
            category = encoder.inverse_transform([prediction])[0]

        # Map category to UI format with emojis if needed
            category_mapping = {
                "Household": "Household waste 🗑",
                "Paper": "Paper 📄",
                "Cardboard": "Cardboard 📦",
                "Glass": "Glass 🍾",
                 "Green": "Green waste 🌿",
                "Cans": "Cans 🥫",
                "Aluminium": "Aluminium 🧴",
                "Metal": "Metal 🪙",
                "Textiles": "Textiles 👕",
                "Oil": "Oil 🛢",
                "Hazardous": "Hazardous waste ⚠",
                "Foam packaging": "Foam packaging ☁"
            }

            ui_category = category_mapping.get(category, category)

            if confidence < 0.3:
                return "Unknown 🚫", float(confidence)

            return ui_category, float(confidence)

        except Exception as e:
            logger.error(f"Error in ML text prediction: {str(e)}")
            # Fall back to rule-based
            return rule_based_prediction(description)
    else:
        # Fall back to rule-based prediction
        logger.info("ML model not available, using rule-based prediction")
        return rule_based_prediction(description)

# Enhanced predict_from_image function with fallback
def predict_from_image(img, model=None, class_names=None):
    """Predict waste type from image with fallback to color-based"""
    if model is None or class_names is None:
        # Fallback to color-based prediction
        logger.info("Image model not available, using color-based prediction")
        return simple_image_prediction(img)
        
    try:
        # Ensure TensorFlow is imported
        import tensorflow as tf
        from tensorflow.keras.preprocessing import image as keras_image
        
        # Preprocess image
        img = img.resize((224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Make prediction
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Get class name
        if class_idx < len(class_names):
            category = class_names[class_idx]
            return category, confidence
        else:
            logger.error(f"Invalid class index: {class_idx}, max expected: {len(class_names)-1}")
            return simple_image_prediction(img)
            
    except Exception as e:
        logger.error(f"Error in image prediction: {str(e)}")
        return simple_image_prediction(img)

# Function to convert waste type selected in UI to API format
# Add this improved convert_waste_type_to_api function to app.py
# Replace the existing function with this one

def convert_waste_type_to_api(ui_waste_type):
    """
    Convert waste type selected in UI to API format with improved handling of emoji and exact matching.
    """
    # First strip any emoji from the waste type if present
    clean_waste_type = ui_waste_type
    # Remove emoji and additional text if present
    if " " in clean_waste_type:
        clean_waste_type = clean_waste_type.split(" ")[0]
    
    mapping = {
        "Household": "Kehricht",
        "Paper": "Papier",
        "Cardboard": "Karton",
        "Glass": "Glas",
        "Green": "Grüngut",
        "Cans": "Dosen",
        "Aluminium": "Aluminium",
        "Metal": "Altmetall",
        "Textiles": "Alttextilien",
        "Oil": "Altöl",
        "Hazardous": "Sonderabfall",
        "Foam": "Styropor"
    }
    
    # Also support full names with emoji
    full_mapping = {
        "Household waste 🗑": "Kehricht",
        "Paper 📄": "Papier",
        "Cardboard 📦": "Karton",
        "Glass 🍾": "Glas",
        "Green waste 🌿": "Grüngut",
        "Cans 🥫": "Dosen",
        "Aluminium 🧴": "Aluminium",
        "Metal 🪙": "Altmetall",
        "Textiles 👕": "Alttextilien",
        "Oil 🛢": "Altöl",
        "Hazardous waste ⚠": "Sonderabfall",
        "Foam packaging ☁": "Styropor"
    }
    
    # Try full match first
    if ui_waste_type in full_mapping:
        return full_mapping[ui_waste_type]
    
    # Then try clean waste type
    if clean_waste_type in mapping:
        return mapping[clean_waste_type]
    
    # If still not found, try case-insensitive partial matching
    ui_waste_lower = ui_waste_type.lower()
    for key, value in full_mapping.items():
        if key.lower() in ui_waste_lower or ui_waste_lower in key.lower():
            return value
    
    # If we get here, return the original input as fallback
    # This might happen if the waste type was already in API format
    return ui_waste_type

# Convert API waste type to UI format (with emojis)
def convert_api_to_ui_waste_type(api_waste_type):
    mapping = {
        "Kehricht": "Household waste 🗑",
        "Papier": "Paper 📄",
        "Karton": "Cardboard 📦",
        "Glas": "Glass 🍾",
        "Grüngut": "Green waste 🌿",
        "Dosen": "Cans 🥫",
        "Aluminium": "Aluminium 🧴",
        "Altmetall": "Metal 🪙",
        "Alttextilien": "Textiles 👕",
        "Altöl": "Oil 🛢",
        "Sonderabfall": "Hazardous waste ⚠",
        "Styropor": "Foam packaging ☁"
    }
    return mapping.get(api_waste_type, api_waste_type)

# Define image class names (needed across pages)
IMAGE_CLASS_NAMES = [
    "Aluminium 🧴",
    "Cans 🥫",
    "Cardboard 📦",
    "Foam packaging ☁",
    "Glass 🍾",
    "Green waste 🌿",
    "Hazardous waste ⚠",
    "Household waste 🗑",
    "Metal 🪙",
    "Oil 🛢",
    "Paper 📄",
    "Plastic ♳",
    "Textiles 👕"
]


def convert_to_ui_label(category):
    mapping = {
        "Aluminium": "Aluminium 🧴",
        "Cans": "Cans 🥫",
        "Cardboard": "Cardboard 📦",
        "Foam packaging": "Foam packaging ☁",
        "Glass": "Glass 🍾",
        "Green waste": "Green waste 🌿",
        "Hazardous": "Hazardous waste ⚠",
        "Household": "Household waste 🗑",
        "Metal": "Metal 🪙",
        "Oil": "Oil 🛢",
        "Paper": "Paper 📄",
        "Plastic": "Plastic",
        "Textiles": "Textiles 👕"
    }
    return mapping.get(category, category)



# Try importing folium and streamlit_folium upfront
try:
    import folium
    from streamlit_folium import st_folium
    logger.info("Successfully imported folium and streamlit_folium")
except ImportError as e:
    logger.warning(f"Failed to import map libraries: {str(e)}")
    logger.warning("Maps functionality might be limited")

# Redirect to the Home page
import streamlit.web.bootstrap
from streamlit.web.server.server import Server
import os

if __name__ == "__main__":
    # Set page config for the main app (needed even though we redirect)
    st.set_page_config(
        page_title="WasteWise - Your smart recycling assistant",
        page_icon="♻️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Hide ALL built-in Streamlit navigation elements - PUT THE CSS HERE
    hide_streamlit_style = """
    <style>
    /* Hide the default sidebar navigation */
    [data-testid="stSidebarNavItems"] {
        display: none !important;
    }

    /* Hide the expand/collapse arrow */
    button[kind="header"] {
        display: none !important;
    }

    /* Remove the extra padding at the top of sidebar */
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem !important;
    }

    /* Optional: Hide app name from sidebar header if present */
    .sidebar-content .sidebar-collapse-control {
        display: none !important;
    }
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Create a simple loading page
    st.write("# Loading WasteWise...")
    st.write("Please wait while we load the application...")

# Then try the switch
    st.switch_page("pages/1_Home.py")