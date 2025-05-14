#imports the necessary libraries 
import streamlit as st # to build the web app
import pandas as pd # to handle data
import numpy as np # for numerical operations
from PIL import Image # for image processing
import requests # to make API calls
import json # to handle JSON data
import pickle # to load ML models
from datetime import datetime # to handle dates
import os # to file system operations
import sys # to handle system operations
import logging # to use app wide logging

# sets up the logging configuration
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# imports helper functions for geolocations, waste disposal, APIs, and maps
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

# initializes variables in session state
if 'waste_info_results' not in st.session_state:
    st.session_state.waste_info_results = None
    st.session_state.show_results = False

# defines a function to check if the ML models exist locally 
def check_ml_models_available():
    """Check if ML model files exist"""
    required_files = ['waste_classifier.pkl', 'waste_vectorizer.pkl', 'waste_encoder.pkl']
    
    for file in required_files:
        if not os.path.exists(file):
            return False
    return True

# defines a function to check if Tensorflow is available
def check_tensorflow_available():
    """Check if TensorFlow is available"""
    try:
        import tensorflow
        return True
    except ImportError:
        return False

# defines a function to check if the text classification model is available and returns an error message if not
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
    
# defines a function to check if the model exists and is available and attempts to download it and returns an error message if not
def download_model_from_reliable_source(model_path):
    """Download model from a reliable source"""
    try:
        if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
            logger.info(f"Model file already exists at {model_path}")
            return True
            
        # URL to your hosted model file
        model_url = "https://github.com/Arnaud-Alvs/CS-Group-6/releases/download/V-1.0.0/waste_image_classifier.h5"       
        
        logger.info(f"Downloading model from {model_url}")
        response = requests.get(model_url, stream=True)
        if response.status_code != 200:
            logger.error(f"Failed to download model: HTTP {response.status_code}")
            return False
            
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"Starting download, expected size: {total_size} bytes")       
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Write the content to file
        with open(model_path, 'wb') as f:
            downloaded_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # Log progress every 10MB
                    if downloaded_size % (10 * 1024 * 1024) == 0:
                        progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
                        logger.info(f"Downloaded {downloaded_size} bytes ({progress:.1f}%)")
        
        # Verify the download
        if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
            actual_size = os.path.getsize(model_path)
            logger.info(f"Model downloaded successfully: {actual_size} bytes")
            return True
        else:
            actual_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
            logger.error(f"Download failed or file too small: {actual_size} bytes")
            if os.path.exists(model_path):
                os.remove(model_path)  # Remove incomplete file
            return False
            
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        if os.path.exists(model_path):
            os.remove(model_path)  # Remove incomplete file if it exists
        return False


# defines a function to check if the image classification model is available and attempts to download it and returns an error message if not
@st.cache_resource
def load_image_model():
    """Load image classification model - focused on reliability"""
    try:
        if not check_tensorflow_available():
            logger.warning("TensorFlow not available")
            return None
            
        import tensorflow as tf
        
        # Define the path to the model (in the current directory)
        model_path = "waste_image_classifier.h5"
        
        # Download the model if it doesn't exist or is too small
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
            logger.info(f"Model not found or too small at {model_path}, downloading...")
            success = download_model_from_reliable_source(model_path)
            if not success:
                logger.error("Failed to download model")
                return None
        else:
            logger.info(f"Model already exists at {model_path}")
        
        # Load the model from the local path
        try:
            logger.info(f"Loading model from {model_path}")
            model = tf.keras.models.load_model(model_path, compile=False)
            logger.info("Model loaded successfully!")
            
            # Verify model structure
            logger.info(f"Model input shape: {model.input_shape}")
            logger.info(f"Model output shape: {model.output_shape}")
            
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # If loading fails, try to delete and re-download
            if os.path.exists(model_path):
                logger.info("Removing corrupted model file and attempting re-download...")
                os.remove(model_path)
                success = download_model_from_reliable_source(model_path)
                if success:
                    try:
                        model = tf.keras.models.load_model(model_path, compile=False)
                        logger.info("Model loaded successfully after re-download!")
                        return model
                    except Exception as e2:
                        logger.error(f"Error loading model after re-download: {str(e2)}")
            return None
            
    except Exception as e:
        logger.error(f"Unexpected error in load_image_model: {str(e)}")
        return None

# Enhanced predict_from_text function with fallback
# Enhanced predict_from_text function without fallback
def predict_from_text(description, model=None, vectorizer=None, encoder=None):
    """Predict waste type from text"""
    if not description:
        return "Unknown 🚫", 0.0
    
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
            return "Unknown 🚫", 0.0
    else:
        # If model is not available, return generic message
        logger.info("ML model not available for text prediction")
        return "Unknown 🚫", 0.0

# Enhanced predict_from_image function with fallback
# Enhanced predict_from_image function without fallback
def predict_from_image(img, model=None, class_names=None):
    """Predict waste type from image using the trained model"""
    if model is None:
        logger.warning("Image model not available")
        return "Unknown 🚫", 0.0
        
    try:
        # Ensure TensorFlow is imported
        import tensorflow as tf
        from tensorflow.keras.preprocessing import image as keras_image
        
        # Preprocess image exactly as during training
        img = img.resize((224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize pixel values
        
        # Make prediction
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Use the MODEL_CLASS_NAMES that match the training
        if class_idx < len(MODEL_CLASS_NAMES):
            # Get the category from training names
            category = MODEL_CLASS_NAMES[class_idx]
            
            # Convert to UI format with emoji using the existing function
            ui_category = convert_to_ui_label(category)
            
            return ui_category, confidence
        else:
            logger.error(f"Invalid class index: {class_idx}, max expected: {len(MODEL_CLASS_NAMES)-1}")
            return "Unknown 🚫", 0.0
            
    except Exception as e:
        logger.error(f"Error in image prediction: {str(e)}")
        return "Unknown 🚫", 0.0
# defines a function to convert the waste type selected in the UI to API format with the improved handling of emojis and exact matching 
def convert_waste_type_to_api(ui_waste_type):
    """
    Convert waste type selected in UI to API format with improved handling of emoji and exact matching.
    """
    # begins with a clean versions of the waste type
    clean_waste_type = ui_waste_type
    # removes any emojis from the waste type 
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
    
    # also includes a full mapping with emojis for better matching
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
    
    # tries the full mapping first for exact matches
    if ui_waste_type in full_mapping:
        return full_mapping[ui_waste_type]
    
    # then tries the clean mapping for exact matches 
    if clean_waste_type in mapping:
        return mapping[clean_waste_type]
    
    # if the wast type is still not found, tries a case-sensitive match 
    ui_waste_lower = ui_waste_type.lower()
    for key, value in full_mapping.items():
        if key.lower() in ui_waste_lower or ui_waste_lower in key.lower():
            return value
    
    # if the waste type is still not found, returns the original waste type
    return ui_waste_type

# converts the API term to a friendly UI format with emojis
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

# defines the image classes that are needed accross all pages for the image recognition 
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

# defines a function to format the waste type into a friendly UI label with emojis
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
        "Plastic": "Plastic ♳",
        "Textiles": "Textiles 👕"
    }
    return mapping.get(category, category)


# tries to imports mapping libaries and handle the errors if they are not available
try:
    import folium
    from streamlit_folium import st_folium
    logger.info("Successfully imported folium and streamlit_folium")
except ImportError as e:
    logger.warning(f"Failed to import map libraries: {str(e)}")
    logger.warning("Maps functionality might be limited")

# imports the stramlit web server and bootstrap modules 
import streamlit.web.bootstrap
from streamlit.web.server.server import Server
import os

# imports the app start page and navigation module 
if __name__ == "__main__":
    # sets the page configuration for the main app
    st.set_page_config(
        page_title="WasteWise - Your smart recycling assistant",
        page_icon="♻️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # hides the default sidebar navigation and the expand collapse arrow using CSS 
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
    
    # creates a loading spinner to show during the loading of the app
    st.write("# Loading WasteWise...")
    st.write("Please wait while we load the application...")

# tries to switch to main page of the app
    st.switch_page("pages/1_Home.py")