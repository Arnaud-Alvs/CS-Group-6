
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

# Function to load the image model
@st.cache_resource
def load_image_model():
    """Load image classification model with detailed error handling"""
    try:
        if not check_tensorflow_available():
            logger.warning("TensorFlow not available")
            return None

        from tensorflow.keras.models import load_model
        import os
        import tensorflow as tf  # Import TF to check version
        
        # Log TensorFlow version for debugging
        logger.info(f"TensorFlow version: {tf.__version__}")
        
        # Get the absolute path to the model file
        model_path = os.path.join(os.path.dirname(__file__), "waste_image_classifier.h5")
        
        # Check if file exists and log its size
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            logger.info(f"Model file exists at {model_path}, size: {file_size} bytes")
            
            # Check if file is empty or very small (likely corrupted)
            if file_size < 1000:  # Less than 1KB
                logger.error(f"Model file appears to be too small ({file_size} bytes), likely corrupted")
                # Delete the corrupted file so we can try downloading again
                os.remove(model_path)
                logger.info(f"Deleted corrupted model file at {model_path}")
        else:
            logger.info(f"Model file does not exist at {model_path}, will download")
        
        # If model doesn't exist or was corrupted, download it
        if not os.path.exists(model_path):
            st.info("Downloading image classification model... This may take a moment.")
            
            # Replace with your actual GitHub release URL
            model_url = "https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0.0/waste_image_classifier.h5"
            
            try:
                import requests
                r = requests.get(model_url, stream=True, timeout=60)
                
                if r.status_code == 200:
                    # Log response headers for debugging
                    logger.info(f"Download headers: {r.headers}")
                    total_size = int(r.headers.get('content-length', 0))
                    logger.info(f"Expected download size: {total_size} bytes")
                    
                    with open(model_path, 'wb') as f:
                        block_size = 1024 * 1024  # 1MB chunks
                        downloaded = 0
                        
                        # Create a progress bar for the download
                        progress_bar = st.progress(0)
                        
                        for chunk in r.iter_content(chunk_size=block_size):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    progress = min(downloaded / total_size, 1.0)
                                    progress_bar.progress(progress)
                                    logger.info(f"Download progress: {progress:.2%}")
                    
                    # Verify the download
                    if os.path.exists(model_path):
                        final_size = os.path.getsize(model_path)
                        logger.info(f"Download complete. File size: {final_size} bytes")
                        
                        if final_size != total_size and total_size > 0:
                            logger.warning(f"Downloaded file size ({final_size}) doesn't match expected size ({total_size})")
                    
                    st.success("Model downloaded successfully!")
                else:
                    st.error(f"Failed to download model: HTTP {r.status_code}")
                    logger.error(f"Failed to download model: HTTP {r.status_code}, Response: {r.text}")
                    return None
            except Exception as e:
                st.error(f"Error downloading model: {str(e)}")
                logger.error(f"Error downloading model: {str(e)}", exc_info=True)
                return None
        
        # Try loading with custom options to handle potential file format issues
        try:
            logger.info("Attempting to load model with standard load_model")
            model = load_model(model_path)
            logger.info("Model loaded successfully with standard method")
            return model
        except Exception as e:
            logger.error(f"Failed to load with standard method: {str(e)}")
            
            # Try alternate loading method with custom options
            try:
                logger.info("Attempting alternate loading method with custom options")
                model = load_model(
                    model_path,
                    compile=False,  # Try without compiling
                    custom_objects=None,  # No custom layers
                )
                logger.info("Model loaded successfully with alternate method")
                return model
            except Exception as e2:
                logger.error(f"Failed with alternate method too: {str(e2)}")
                
                # If all loading attempts fail, delete the file so we can try again next time
                if os.path.exists(model_path):
                    os.remove(model_path)
                    logger.info(f"Deleted potentially corrupted model file at {model_path}")
                
                st.error("Unable to load the model file. It may be corrupted or incompatible with this version of TensorFlow.")
                return None

    except Exception as e:
        logger.error(f"Error in load_image_model: {str(e)}", exc_info=True)
        return None
# Rules-based fallback prediction when ML models aren't available
def rule_based_prediction(description):
    """Rule-based prediction for when ML models aren't available"""
    description = description.lower()
    
    # Keywords for each category
    keywords = {
        "Household waste 🗑": ["trash", "garbage", "waste", "dirty", "leftover", "broken", "ordinary"],
        "Paper 📄": ["paper", "newspaper", "magazine", "book", "printer", "envelope", "document"],
        "Cardboard 📦": ["cardboard", "carton", "box", "packaging", "thick paper"],
        "Glass 🍾": ["glass", "bottle", "jar", "container", "mirror", "window"],
        "Green waste 🌿": ["green", "grass", "leaf", "leaves", "plant", "garden", "flower", "vegetable", "fruit"],
        "Cans 🥫": ["can", "tin", "aluminum can", "soda", "drink can", "food can"],
        "Aluminium 🧴": ["aluminum", "foil", "tray", "container", "lid", "wrap", "packaging"],
        "Metal 🪙": ["metal", "iron", "steel", "scrap", "nails", "screws", "wire"],
        "Textiles 👕": ["textile", "clothes", "fabric", "shirt", "pants", "cloth", "cotton", "wool"],
        "Oil 🛢": ["oil", "cooking oil", "motor oil", "lubricant", "grease"],
        "Hazardous waste ⚠": ["battery", "chemical", "toxic", "medicine", "paint", "solvent", "cleaner"],
        "Foam packaging ☁": ["foam", "styrofoam", "polystyrene", "packing", "cushion", "insulation"]
    }
    
    # Score each category
    scores = {}
    for category, word_list in keywords.items():
        scores[category] = 0
        for word in word_list:
            if word in description:
                scores[category] += 1
    
    # Find best category
    if any(scores.values()):
        best_category = max(scores, key=scores.get)
        confidence = min(0.7, scores[best_category] / len(keywords[best_category]))
        return best_category, confidence
    else:
        return "Household waste 🗑", 0.3  # Default category

# Simple image-based prediction as fallback
def simple_image_prediction(image):
    """Simple color-based prediction as fallback for image classification"""
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Analyze average color
        avg_color = np.mean(img_array, axis=(0, 1))
        
        # Simple logic based on dominant color
        r, g, b = avg_color[:3]
        
        if g > r and g > b:  # Green dominant
            return "Green waste 🌿", 0.5
        elif b > r and b > g:  # Blue dominant
            return "Paper 📄", 0.4
        elif r > g and r > b:  # Red/Brown dominant
            return "Cardboard 📦", 0.4
        elif r > 200 and g > 200 and b > 200:  # Very light
            return "Foam packaging ☁", 0.4
        elif r < 50 and g < 50 and b < 50:  # Very dark
            return "Metal 🪙", 0.4
        else:
            return "Household waste 🗑", 0.3
    except Exception as e:
        logger.error(f"Error in color-based prediction: {str(e)}")
        return "Household waste 🗑", 0.3

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