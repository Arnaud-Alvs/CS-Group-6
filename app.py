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
from streamlit_folium import st_folium
import folium

# Configure error handling and logging
import logging
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
        create_interactive_map  # Add the new function here
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

# Page configuration
st.set_page_config(
    page_title="WasteWise - Your smart recycling assistant",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title
st.title("WasteWise - Your smart recycling assistant")
st.markdown("### Easily find where to dispose of your waste and contribute to a cleaner environment")

if 'waste_info_results' not in st.session_state:
    st.session_state.waste_info_results = None
    st.session_state.show_results = False


# Function to check if ML models are available
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
    """Load image classification model with proper error handling"""
    try:
        if not check_tensorflow_available():
            logger.warning("TensorFlow not available")
            return None
            
        if not os.path.exists("waste_image_classifier.h5"):
            logger.warning("Image model file not found")
            return None
            
        from tensorflow.keras.models import load_model
        return load_model("waste_image_classifier.h5")
    except Exception as e:
        logger.error(f"Error loading image model: {str(e)}")
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
def convert_waste_type_to_api(ui_waste_type):
    mapping = {
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
    return mapping.get(ui_waste_type, ui_waste_type)

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

# Main interface with tabs
tab1, tab2, tab3 = st.tabs(["Find a collection point", "Identify waste", "About"])

with tab1:
    st.header("Find Collection Information")

    st.markdown(
        """
        Welcome to WasteWise! Enter the type of waste you want to dispose of
        and your address in St. Gallen to find nearby collection points and
        upcoming collection dates.
        """
    )
    # --- User Input Section ---
    # Get available waste types from location_api
    available_waste_types_german = get_available_waste_types()
    # Translate waste types for the dropdown
    available_waste_types_english = [translate_waste_type(wt) for wt in available_waste_types_german]

    # Create a mapping from English back to German for API calls
    waste_type_mapping = dict(zip(available_waste_types_english, available_waste_types_german))

    # Button to trigger the search
 # In app.py where you handle the "Find Information" button click and display results:

with st.form(key="waste_search_form"):
        # Move your input fields here
        selected_waste_type_english = st.selectbox(
            "Select Waste Type:",
            options=available_waste_types_english,
            help="Choose the type of waste you want to dispose of."
        )

        user_address = st.text_input(
            "Enter your Address in St. Gallen:",
            placeholder="e.g., Musterstrasse 1",
            help="Enter your address, it must include a street name and number."
        )
        
        # Form submit button
        submit_button = st.form_submit_button("Find Information")
    
    # Process form submission
if submit_button:
        if not user_address:
            st.warning("Please enter your address.")
        elif not selected_waste_type_english:
            st.warning("Please select a waste type.")
        else:
            # Show a progress indicator
            with st.spinner(f"Searching for disposal options for {selected_waste_type_english}..."):
                # Translate selected waste type back to German for API calls
                selected_waste_type_german = waste_type_mapping.get(selected_waste_type_english, selected_waste_type_english)
                
                # Store inputs in session state
                st.session_state.selected_waste_type = selected_waste_type_english
                st.session_state.user_address = user_address
                
                # Use the combined function for waste disposal information
                waste_info = handle_waste_disposal(user_address, selected_waste_type_german)
                
                # Store results in session state
                st.session_state.waste_info_results = waste_info
                st.session_state.show_results = True
    
    # Display results section - only if we have data
if 'show_results' in st.session_state and st.session_state.show_results:
        # Clear separation from the input form
        st.markdown("---")
        
        # Create a container for the results that won't be affected by other changes
        results_container = st.container()
        
        with results_container:
            waste_info = st.session_state.waste_info_results
            selected_waste_type_english = st.session_state.selected_waste_type
            user_address = st.session_state.user_address
            
            st.subheader("Search Results")
            st.markdown(waste_info["message"])
            
            # Display collection points and map in separate columns if available
            if waste_info["has_disposal_locations"]:
                st.markdown(f"### Nearest Collection Points for {selected_waste_type_english}")
                
                # Get user coordinates for the map
                user_coords = get_coordinates(user_address)
                
                # Create a 2-column layout
                if user_coords:
                    map_col, info_col = st.columns([3, 2])
                    
                    with map_col:
                        st.caption("Hover over markers for info, click for details.")
                        
                        # Create the map in its own container
                        interactive_map = create_interactive_map(user_coords, waste_info["collection_points"])
                        
                        # Use the fixed st_folium call
                        st_folium(
                            interactive_map, 
                            width=None,  # Full width
                            height=400,
                            returned_objects=[],  # This prevents reruns
                            key=f"map_{user_address}_{selected_waste_type_english}"
                        )
                    
                    with info_col:
                        st.markdown("**Nearest Locations**")
                        for i, point in enumerate(waste_info["collection_points"][:5]):  # Show top 5
                            with st.expander(f"{point['name']} ({point['distance']:.2f} km)"):
                                st.markdown(f"**Accepted Waste:** {', '.join([translate_waste_type(wt) for wt in point['waste_types']])}")
                                if point['opening_hours'] and point['opening_hours'] != "N/A":
                                    st.markdown(f"**Opening Hours:** {point['opening_hours']}")
            
            # Display collection date if available
            if waste_info["has_scheduled_collection"]:
                st.markdown(f"### Next Collection Date for {selected_waste_type_english}")
                next_collection = waste_info["next_collection_date"]
                
                # Create a nice collection date display
                collection_date = next_collection['date'].strftime('%A, %B %d, %Y')
                collection_time = next_collection.get('time', '')
                
                # Use a success message for the date
                date_html = f"""
                <div style="background-color: #d4edda; border-radius: 5px; padding: 15px; margin-bottom: 15px;">
                    <h3 style="color: #155724; margin-top: 0;">Next Collection: {collection_date}</h3>
                    {f"<p style='margin-bottom: 0;'><strong>Time:</strong> {collection_time}</p>" if collection_time and collection_time != "N/A" else ""}
                    {f"<p style='margin-bottom: 0;'><strong>Details:</strong> {next_collection['description']}</p>" if next_collection['description'] and next_collection['description'] != "Collection" else ""}
                    {f"<p style='margin-bottom: 0;'><strong>Area:</strong> {next_collection['area']}</p>" if next_collection['area'] and next_collection['area'] != "N/A" else ""}
                </div>
                """
                st.markdown(date_html, unsafe_allow_html=True)
            
            # Add a "Search Again" button at the bottom of results
            if st.button("Search Again", key="search_again"):
                # Clear the results
                st.session_state.show_results = False
                st.experimental_rerun()
    # Separator
st.markdown("---")

    # --- General Information Section ---
st.header("General Waste Information")

    # Tip of the day
st.subheader("Tip of the Day")
tips_of_the_day = [
        "Recycling one aluminum can saves enough energy to run a TV for three hours.",
        "Paper can be recycled up to 7 times before the fibers become too short.",
        "Glass is 100% recyclable and can be recycled infinitely without losing its quality!",
        "A mobile phone contains more than 70 different materials, many of which are recyclable.",
        "Batteries contain toxic heavy metals, never throw them away with household waste.",
        "Consider putting a 'No Junk Mail' sticker on your mailbox to reduce paper waste.",
        "Composting can reduce the volume of your household waste by up to 30%.",
        "Remember to break down cardboard packaging before disposing of it to save space.",
        "LED bulbs are less harmful to the environment and last longer."
    ]
import random
st.info(random.choice(tips_of_the_day))
  
    # Separator
st.markdown("---")
    # Useful links (keeping the existing links, replace example.com with actual links if available)
st.subheader("Useful links")
st.markdown("[Complete recycling guide](https://www.stadt.sg.ch/home/umwelt-energie/entsorgung.html)") # Example link
st.markdown("[Reducing waste in everyday life](https://www.bafu.admin.ch/bafu/en/home/topics/waste/guide-to-waste-a-z/avoiding-waste.html)") # Example link
st.markdown("[Waste legislation in Switzerland](https://www.bafu.admin.ch/bafu/en/home/topics/waste/legal-basis.html)") # Example link
st.markdown("[Official St. Gallen city website](https://www.stadt.sg.ch/)")

with tab2:
    st.header("Identify your waste")
    
    # Check if ML models are available and load them once
    text_model, text_vectorizer, text_encoder = load_text_model()
    image_model = load_image_model()
    
    # Define consistent class names exactly matching training data
    image_class_names = [
        "Household waste 🗑", 
        "Paper 📄", 
        "Cardboard 📦", 
        "Glass 🍾", 
        "Green waste 🌿", 
        "Cans 🥫", 
        "Aluminium 🧴", 
        "Foam packaging ☁", 
        "Metal 🪙", 
        "Textiles 👕", 
        "Oil 🛢", 
        "Hazardous waste ⚠"
    ]
    
    # Show model status
    col1, col2 = st.columns(2)
    with col1:
        if text_model is not None:
            st.success("✅ Text analysis available")
        else:
            st.warning("⚠️ Text analysis: Using rule-based fallback")
            
    with col2:
        if image_model is not None:
            st.success("✅ Image analysis available")
        else:
            st.warning("⚠️ Image analysis: Using color-based fallback")
            
    # Input from user
    waste_description = st.text_area("Describe your waste (material, size, usage, etc.)")
    uploaded_file = st.file_uploader("Or upload a photo of your waste", type=["jpg", "jpeg", "png"])
    
    # Display uploaded image
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", width=300)
        except Exception as e:
            st.error(f"Error opening image: {e}")
            image = None
    else:
        image = None
    
    # Analyze button
    if st.button("Identify"):
        if waste_description or (uploaded_file and image is not None):
            with st.spinner("Analyzing your waste..."):
                
                results = []
                
                # Text analysis
                if waste_description:
                    category, confidence = predict_from_text(
                        waste_description, 
                        model=text_model, 
                        vectorizer=text_vectorizer, 
                        encoder=text_encoder
                    )
                    if category:
                        results.append((category, confidence, "text"))
                        st.success(f"Text analysis result: {category} (confidence: {confidence:.2%})")
                
                # Image analysis
                if uploaded_file and image is not None:
                    category, confidence = predict_from_image(
                        image, 
                        model=image_model, 
                        class_names=image_class_names
                    )
                    if category:
                        results.append((category, confidence, "image"))
                        st.success(f"Image analysis result: {category} (confidence: {confidence:.2%})")
                
                # Show the most confident result or combine results
                if len(results) > 1:
                    # Take the result with highest confidence
                    final_result = max(results, key=lambda x: x[1])
                    category, confidence, method = final_result
                    st.info(f"Final recommendation: {category} (based on {method} analysis)")
                elif len(results) == 1:
                    category = results[0][0]
                else:
                    st.error("Analysis failed. Please try again.")
                    category = None
                    
                # Show sorting advice if we have a category
                if category:
                    # Show sorting advice for the predicted category
                    st.subheader("Waste sorting advice")
                    
                    # Extract the base category without emoji
                    base_category = category.split(" ")[0] if " " in category else category
                    
                    sorting_advice = {
                        "Household": {
                            "bin": "General waste bin (gray/black)",
                            "tips": [
                                "Ensure waste is properly bagged",
                                "Remove any recyclable components first",
                                "Compact waste to save space"
                            ]
                        },
                        "Paper": {
                            "bin": "Paper recycling (blue)",
                            "tips": [
                                "Remove any plastic components or covers",
                                "Flatten to save space",
                                "Keep dry and clean"
                            ]
                        },
                        "Cardboard": {
                            "bin": "Cardboard recycling (blue/brown)",
                            "tips": [
                                "Break down boxes to save space",
                                "Remove tape and plastic parts",
                                "Keep dry and clean"
                            ]
                        },
                        "Glass": {
                            "bin": "Glass container (green/clear/brown)",
                            "tips": [
                                "Separate by color if required",
                                "Remove caps and lids",
                                "Rinse containers before disposal"
                            ]
                        },
                        "Green": {
                            "bin": "Organic waste (green/brown)",
                            "tips": [
                                "No meat or cooked food in some systems",
                                "No plastic bags, even biodegradable ones",
                                "Cut large branches into smaller pieces"
                            ]
                        },
                        "Cans": {
                            "bin": "Metal recycling",
                            "tips": [
                                "Rinse containers before recycling",
                                "Crush if possible to save space",
                                "Labels can typically stay on"
                            ]
                        },
                        "Aluminium": {
                            "bin": "Metal recycling",
                            "tips": [
                                "Clean off food residue",
                                "Can be crushed to save space",
                                "Collect smaller pieces together"
                            ]
                        },
                        "Metal": {
                            "bin": "Metal recycling or collection point",
                            "tips": [
                                "Larger items may need special disposal",
                                "Remove non-metal components if possible",
                                "Take to recycling center if too large"
                            ]
                        },
                        "Textiles": {
                            "bin": "Textile collection bins",
                            "tips": [
                                "Clean and dry items only",
                                "Pair shoes together",
                                "Separate for donation vs. recycling"
                            ]
                        },
                        "Oil": {
                            "bin": "Special collection point",
                            "tips": [
                                "Never pour down the drain",
                                "Keep in original container if possible",
                                "Take to recycling center or garage"
                            ]
                        },
                        "Hazardous": {
                            "bin": "Hazardous waste collection",
                            "tips": [
                                "Keep in original container if possible",
                                "Never mix different chemicals",
                                "Take to special collection points"
                            ]
                        },
                        "Foam": {
                            "bin": "Special recycling or general waste",
                            "tips": [
                                "Check local rules as they vary widely",
                                "Some recycling centers accept clean foam",
                                "Break into smaller pieces"
                            ]
                        }
                    }
                    
                    # Find matching advice
                    for key, advice in sorting_advice.items():
                        if key in base_category:
                            st.write(f"**Disposal bin:** {advice['bin']}")
                            st.write("**Tips:**")
                            for tip in advice['tips']:
                                st.write(f"- {tip}")
                            break
                    else:
                        # Default advice if no match
                        st.write("Please check your local waste management guidelines for this specific item.")
                    
                    # Offer to search for collection points
                    st.markdown("---")
                    if st.button("Find collection points for this waste type"):
                        # Store waste type in session state
                        if "session_state" not in st.session_state:
                            st.session_state["waste_type"] = category
                            st.session_state["active_tab"] = 0  # Find a collection point tab
                        st.experimental_rerun()
        
        else:
            st.error("Please describe your waste or upload an image")

with tab3:
    st.header("About WasteWise")
    
    # System status
    st.subheader("System Status")
    
    # Check API connectivity
    try:
        test_url = "https://daten.stadt.sg.ch"
        response = requests.get(test_url, timeout=5)
        if response.status_code == 200:
            st.success("✅ St. Gallen API: Connected")
        else:
            st.warning(f"⚠️ St. Gallen API: Status code {response.status_code}")
    except Exception as e:
        st.error(f"❌ St. Gallen API: Disconnected - {str(e)}")
    
    # Check ML models
    if check_ml_models_available():
        st.success("✅ Text Classification Model: Available")
    else:
        st.warning("⚠️ Text Classification Model: Not found (using rule-based fallback)")
    
    if os.path.exists("waste_image_classifier.h5"):
        if check_tensorflow_available():
            st.success("✅ Image Classification Model: Available")
        else:
            st.warning("⚠️ Image Classification Model: File exists but TensorFlow not installed (using color-based fallback)")
    else:
        st.warning("⚠️ Image Classification Model: Not found (using color-based fallback)")
    
    # Project description
    st.markdown("""
    ## Our mission
    
    **WasteWise** is an educational application developed as part of a university project. 
    Our mission is to simplify waste sorting and contribute to better recycling by:
    
    - Helping users correctly identify their waste
    - Providing personalized sorting advice
    - Locating the nearest collection points
    - Informing about upcoming collection dates
    - Raising awareness about the importance of recycling
    
    ## Technologies used
    
    This application is built with:
    
    - **Streamlit**: for the user interface
    - **Machine Learning**: for waste identification
    - **Geolocation API**: to find collection points
    - **Open Data API**: for St. Gallen waste collection data
    - **Data processing**: to analyze and classify waste
    
    ## How it works?
    
    1. **Waste identification**: Our AI system analyzes your description or photo to determine the type of waste.
    
    2. **Personalized advice**: Based on the identified waste type, we provide specific advice for its sorting.
    
    3. **Locating collection points**: We help you find the closest collection points to dispose of your waste.
    
    4. **Collection dates**: We inform you of upcoming collection dates for your waste in your area.
    """)
    
    # Display fictional statistics on application usage
    st.subheader("WasteWise Impact")
    
    # Use columns to display statistics side by side
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Waste identified", value="12,543", delta="1,243 this week")
    
    with col2:
        st.metric(label="Collection points", value="3,879", delta="152 new")
    
    with col3:
        st.metric(label="Active users", value="853", delta="57 new")
    
    # User feedback
    st.subheader("Your feedback matters")
    st.write("Help us improve WasteWise by sharing your suggestions:")
    
    with st.form("feedback_form"):
        feedback_text = st.text_area("Your comments and suggestions")
        user_email = st.text_input("Your email (optional)")
        submit_button = st.form_submit_button("Send my feedback")
        
        if submit_button:
            if feedback_text:
                st.success("Thank you for your feedback! We have received it.")
                # In a real project, you would record this feedback in a database
            else:
                st.error("Please enter a comment before sending.")

# Sidebar with additional features
with st.sidebar:
    st.header("Options")
    
    # Tips of the day
    st.subheader("Tip of the day")
    tips_of_the_day = [
        "Did you know that plastic caps can be recycled separately?",
        "Glass can be recycled infinitely without losing its quality!",
        "A mobile phone contains more than 70 different materials, many of which are recyclable.",
        "Batteries contain toxic heavy metals, never throw them away with household waste.",
        "Consider putting a 'No Junk Mail' sticker on your mailbox to reduce paper waste.",
        "Composting can reduce the volume of your household waste by up to 30%.",
        "Remember to break down cardboard packaging before disposing of it to save space.",
        "LED bulbs are less harmful to the environment and last longer."
    ]
    import random
    st.info(random.choice(tips_of_the_day))
    
    # Separator
    st.markdown("---")
    
    # Useful links
    st.subheader("Useful links")
    st.markdown("[Complete recycling guide](https://example.com)")
    st.markdown("[Reducing waste in everyday life](https://example.com)")
    st.markdown("[Waste legislation in Switzerland](https://example.com)")
    st.markdown("[Official St. Gallen city website](https://www.stadt.sg.ch/)")

# Footer
st.markdown("---")
st.markdown("© 2025 WasteWise - University Project | [Contact](mailto:contact@wastewise.example.com) | [Legal notice](https://example.com)")