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
# Add this function to your app.py
def verify_h5_model_format(model_path):
    """Verify if H5 file has correct format and convert if needed"""
    try:
        import h5py
        import tensorflow as tf
        import os
        
        # Try to open with h5py to check format
        try:
            with h5py.File(model_path, 'r') as f:
                # Check for key Keras components
                if 'model_weights' in f or 'layer_names' in f:
                    logger.info("File appears to be a valid HDF5 Keras model")
                    return True
                else:
                    logger.warning("File is HDF5 but may not be a Keras model")
        except Exception as e:
            logger.error(f"Error checking HDF5 format: {str(e)}")
            
        # If we get here, try to load and re-save the model in a compatible format
        logger.info("Attempting to convert model format...")
        
        # Try loading the model
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Save in SavedModel format (more compatible)
        saved_model_dir = os.path.join(os.path.dirname(model_path), "saved_model")
        model.save(saved_model_dir, save_format="tf")
        logger.info(f"Model converted and saved to {saved_model_dir}")
        
        return saved_model_dir
        
    except Exception as e:
        logger.error(f"Model verification/conversion failed: {str(e)}")
        return False

# Function to load the text model - fixed version
def download_and_convert_model():
    """Download model from Google Drive with Keras 3 compatibility"""
    logger.info("Starting download_and_convert_model function")
    try:
        if not check_tensorflow_available():
            logger.warning("TensorFlow not available")
            return None

        import tensorflow as tf
        import os
        import requests
        
        # Define paths
        h5_model_path = os.path.join(os.path.dirname(__file__), "waste_image_classifier.h5")
        keras_model_path = os.path.join(os.path.dirname(__file__), "waste_image_classifier.keras")
        
        # Check if Keras model already exists
        if os.path.exists(keras_model_path):
            logger.info(f"Keras model exists at {keras_model_path}")
            try:
                # Try to load the existing Keras model
                model = tf.keras.models.load_model(keras_model_path)
                logger.info("Successfully loaded existing Keras model")
                return model
            except Exception as e:
                logger.error(f"Failed to load existing Keras model: {str(e)}")
                # Continue to download and convert
        
        # Download the H5 model if needed
        if not os.path.exists(h5_model_path):
            file_id = "17Uxb4w3ehpK0rNj0crgBT1BD7xvFxByz"
            download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
            
            st.info("Downloading image classification model... This may take a moment.")
            
            session = requests.Session()
            response = session.get(download_url, stream=True)
            token = None
            
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
                    logger.info("Received download token for large file")
                    break
            
            if token:
                params = {'id': file_id, 'confirm': token, 'export': 'download'}
                response = session.get("https://drive.google.com/uc", params=params, stream=True)
            
            total_size = int(response.headers.get('content-length', 0))
            logger.info(f"Starting download, expected size: {total_size} bytes")
            
            progress_bar = st.progress(0)
            downloaded = 0
            
            with open(h5_model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = min(downloaded / total_size, 1.0) if total_size > 0 else 0
                        progress_bar.progress(progress)
            
            if not os.path.exists(h5_model_path) or os.path.getsize(h5_model_path) < 1000000:
                logger.error("Download failed or file too small")
                st.error("Failed to download model file")
                return None
            
            st.success("Model downloaded successfully")
        
        # Try different approaches to load the model
        try:
            # Try direct loading of the H5 model first
            logger.info("Attempting to load H5 model directly")
            try:
                h5_model = tf.keras.models.load_model(h5_model_path, compile=False)
                logger.info("Successfully loaded H5 model directly")
                return h5_model
            except Exception as e1:
                logger.error(f"Failed to load H5 model directly: {str(e1)}")
                
            # If direct loading fails, try loading with custom options
            logger.info("Attempting to load H5 model with custom options")
            try:
                custom_model = tf.keras.models.load_model(
                    h5_model_path,
                    compile=False,
                    custom_objects={}
                )
                logger.info("Successfully loaded H5 model with custom options")
                return custom_model
            except Exception as e2:
                logger.error(f"Failed to load H5 model with custom options: {str(e2)}")
                
            # If that fails, try converting to Keras format (for Keras 3)
            logger.info("Attempting to convert from H5 to Keras format")
            try:
                # Import h5py to check if file is valid
                import h5py
                with h5py.File(h5_model_path, 'r') as h5file:
                    # Just check if file can be opened
                    logger.info(f"H5 file opened, keys: {list(h5file.keys())}")
                
                # Try using the lower-level loader
                from tensorflow.python.keras.saving import saving_utils
                h5_model = saving_utils.load_model_from_hdf5(h5_model_path)
                
                # If we got here, save in new Keras format
                h5_model.save(keras_model_path)
                logger.info(f"Converted H5 model to Keras format at {keras_model_path}")
                return h5_model
            except Exception as e3:
                logger.error(f"Failed to convert to Keras format: {str(e3)}")
            
            # As a last resort, try a direct low-level approach for Keras 3
            logger.info("Attempting last resort approach for Keras 3")
            try:
                # Try direct import approach for Keras 3
                from tensorflow.keras.models import model_from_json
                from tensorflow.keras.models import load_weights
                
                with h5py.File(h5_model_path, 'r') as f:
                    # Get model architecture
                    model_json = f.attrs.get('model_config')
                    if model_json:
                        model_json = model_json.decode('utf-8')
                        # Create model from JSON
                        model = model_from_json(model_json)
                        # Load weights
                        model.load_weights(h5_model_path)
                        logger.info("Successfully loaded model using model_from_json")
                        return model
            except Exception as e4:
                logger.error(f"Failed with last resort approach: {str(e4)}")
                
            # If we've tried everything and still failed, return None
            logger.error("All loading attempts failed")
            return None
            
        except Exception as e:
            logger.error(f"General error in model loading: {str(e)}")
            return None

    except Exception as e:
        logger.error(f"Unexpected error in download_and_convert_model: {str(e)}", exc_info=True)
        return None


# In app.py, modify your load_image_model function to integrate the new approach
@st.cache_resource
def load_image_model():
    """Load image model with robust fallbacks"""
    logger.info("Starting load_image_model function")
    
    # First try the original approach
    try:
        model = download_and_convert_model()
        if model is not None:
            logger.info("Successfully loaded TensorFlow model")
            return model
    except Exception as e:
        logger.error(f"Error loading TensorFlow model: {str(e)}")
    
    # If TensorFlow model fails, use our SimpleImageClassifier instead
    logger.info("Using SimpleImageClassifier as fallback")
    return SimpleImageClassifier()



class SimpleImageClassifier:
    """A simple image classifier using color histograms and hand-crafted features"""
    
    def __init__(self):
        self.class_names = [
            "Aluminium 🧴", "Cans 🥫", "Cardboard 📦", "Foam packaging ☁", 
            "Glass 🍾", "Green waste 🌿", "Hazardous waste ⚠", "Household waste 🗑", 
            "Metal 🪙", "Oil 🛢", "Paper 📄", "Plastic", "Textiles 👕"
        ]
        
    def extract_features(self, image):
        """Extract color histogram and other simple features from image"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Resize image for consistent features
        from PIL import Image
        resized_img = Image.fromarray(img_array).resize((100, 100))
        img_array = np.array(resized_img)
        
        # Extract average RGB values
        avg_color = np.mean(img_array, axis=(0, 1))
        
        # Extract color histograms for each channel
        r_hist, _ = np.histogram(img_array[:,:,0], bins=10, range=(0, 256))
        g_hist, _ = np.histogram(img_array[:,:,1], bins=10, range=(0, 256))
        b_hist, _ = np.histogram(img_array[:,:,2], bins=10, range=(0, 256))
        
        # Extract edge information
        try:
            from scipy import ndimage
            edges_x = ndimage.sobel(img_array[:,:,0], axis=0)
            edges_y = ndimage.sobel(img_array[:,:,0], axis=1)
            edge_magnitude = np.sqrt(edges_x**2 + edges_y**2)
            edge_mean = np.mean(edge_magnitude)
            edge_std = np.std(edge_magnitude)
        except ImportError:
            # If scipy is not available, use dummy values
            edge_mean = 0
            edge_std = 0
        
        # Combine all features
        features = {
            'avg_r': avg_color[0],
            'avg_g': avg_color[1],
            'avg_b': avg_color[2],
            'r_hist': r_hist,
            'g_hist': g_hist,
            'b_hist': b_hist,
            'edge_mean': edge_mean,
            'edge_std': edge_std
        }
        
        return features
    
    def predict(self, image):
        """Predict waste category from image"""
        features = self.extract_features(image)
        
        # Simple rules based on color and texture
        avg_r, avg_g, avg_b = features['avg_r'], features['avg_g'], features['avg_b']
        edge_mean = features['edge_mean']
        
        # Define color rules
        is_green = avg_g > max(avg_r, avg_b) * 1.1
        is_blue = avg_b > max(avg_r, avg_g) * 1.1
        is_red = avg_r > max(avg_g, avg_b) * 1.1
        is_bright = avg_r > 200 and avg_g > 200 and avg_b > 200
        is_dark = avg_r < 60 and avg_g < 60 and avg_b < 60
        is_yellow = avg_r > 200 and avg_g > 200 and avg_b < 100
        is_gray = abs(avg_r - avg_g) < 20 and abs(avg_r - avg_b) < 20 and abs(avg_g - avg_b) < 20
        has_edges = edge_mean > 30
        
        # Define rules for classification
        if is_green:
            category = "Green waste 🌿"
            confidence = 0.6
        elif is_blue and not has_edges:
            category = "Paper 📄"
            confidence = 0.5
        elif is_yellow or (is_red and not has_edges):
            category = "Cardboard 📦"
            confidence = 0.5
        elif is_bright and not has_edges:
            category = "Foam packaging ☁"
            confidence = 0.5
        elif is_dark or (is_gray and has_edges):
            category = "Metal 🪙"
            confidence = 0.5
        elif is_gray and not has_edges:
            category = "Household waste 🗑"
            confidence = 0.4
        elif (avg_r > 130 and avg_g > 70 and avg_g < 120 and avg_b < 80) or (is_red and has_edges):
            category = "Hazardous waste ⚠"
            confidence = 0.4
        elif (avg_b > 150 and avg_g > 150 and avg_r < 100) or (is_blue and has_edges):
            category = "Glass 🍾"
            confidence = 0.5
        elif (avg_r > 160 and avg_g > 120 and avg_b < 100):
            category = "Textiles 👕"
            confidence = 0.4
        elif (avg_r < 50 and avg_g < 50 and avg_b < 50) and has_edges:
            category = "Oil 🛢"
            confidence = 0.4
        elif is_gray and not has_edges:
            category = "Aluminium 🧴"
            confidence = 0.4
        elif (avg_r > 100 and avg_g > 100 and avg_b > 100) and is_gray:
            category = "Cans 🥫"
            confidence = 0.4
        else:
            category = "Household waste 🗑"
            confidence = 0.3
        
        # Confidence is proportional to certainty of the decision
        return category, confidence


def predict_from_image(img, model=None, class_names=None):
    """Predict waste type from image with multiple fallbacks"""
    try:
        # If model is our SimpleImageClassifier, use its predict method
        if model is not None and isinstance(model, SimpleImageClassifier):
            logger.info("Using SimpleImageClassifier for prediction")
            return model.predict(img)
        
        # If model is None or not SimpleImageClassifier, try TensorFlow prediction
        if model is not None and class_names is not None:
            logger.info("Using TensorFlow model for prediction")
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
        
        # Fallback to color-based prediction if everything else fails
        logger.info("Image model not available, using color-based prediction")
        return simple_image_prediction(img)
            
    except Exception as e:
        logger.error(f"Error in image prediction: {str(e)}")
        return simple_image_prediction(img)
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