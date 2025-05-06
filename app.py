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
        get_collection_dates, 
        format_collection_points, 
        get_available_waste_types,
        translate_waste_type,
        COLLECTION_POINTS_ENDPOINT,
        COLLECTION_DATES_ENDPOINT
    )
    logger.info(f"Successfully imported location_api functions")
    logger.info(f"Collection Points API: {COLLECTION_POINTS_ENDPOINT}")
    logger.info(f"Collection Dates API: {COLLECTION_DATES_ENDPOINT}")
except ImportError as e:
    st.error(f"Failed to import from location_api.py: {str(e)}")
    logger.error(f"Import error: {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error when importing location_api.py: {str(e)}")
    logger.error(f"Unexpected error: {str(e)}")
    st.stop()

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

# Function to load the text model
@st.cache_resource
def load_text_model():
    """Load text classification model"""
    try:
        with open('waste_classifier.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('waste_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('waste_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        return model, vectorizer, encoder
    except FileNotFoundError:
        logger.warning("ML model files not found")
        return None, None, None
    except Exception as e:
        logger.error(f"Error loading text model: {str(e)}")
        return None, None, None

# Function to load the image model
@st.cache_resource
def load_image_model():
    """Load image classification model"""
    try:
        # Check if TensorFlow is available
        if not check_tensorflow_available():
            logger.warning("TensorFlow not available")
            return None
            
        # Check if model file exists
        if not os.path.exists("waste_image_classifier.h5"):
            logger.warning("Image model file not found")
            return None
            
        from tensorflow.keras.models import load_model
        return load_model("waste_image_classifier.h5")
    except Exception as e:
        logger.error(f"Error loading image model: {str(e)}")
        return None

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
    st.header("Find where to dispose of your waste")
    
    # Get available waste types from UI format
    ui_waste_types = [
        "Household waste 🗑",
        "Paper 📄",
        "Cardboard 📦",
        "Glass 🍾",
        "Green waste 🌿",
        "Cans 🥫",
        "Aluminium 🧴",
        "Metal 🪙",
        "Textiles 👕",
        "Oil 🛢",
        "Hazardous waste ⚠",
        "Foam packaging ☁"
    ]
    
    # Ask for waste type
    waste_type = st.selectbox(
        "What type of waste do you want to dispose of?",
        ui_waste_types
    )
    
    # Convert selected waste type to API format
    api_waste_type = convert_waste_type_to_api(waste_type)
    
    # Ask for location (with more specific instructions)
    user_location = st.text_input(
        "Enter your address in St. Gallen", 
        placeholder="Example: Bahnhofstrasse 1, 9000 St. Gallen",
        help="Enter your complete address including street name, number, postal code, and city"
    )
    
    # Fixed radius (not visible to user)
    search_radius = 10  # 10 km maximum search radius
    
    if st.button("Search for collection points"):
        if user_location:
            with st.spinner("Searching for collection points..."):
                # Get geographic coordinates of the address
                coordinates = get_coordinates(user_location)
                
                if coordinates:
                    st.info(f"✓ Found your location at coordinates: {coordinates['lat']:.4f}, {coordinates['lon']:.4f}")
                    
                    # Find nearby collection points
                    collection_points = find_collection_points(coordinates, api_waste_type, search_radius)
                    
                    if collection_points:
                        st.success(f"Found {len(collection_points)} collection points for {waste_type} within 10 km of your location")
                        
                        # Display map
                        map_data, points = format_collection_points(collection_points)
                        
                        # Add user location to the map
                        user_marker = pd.DataFrame([{
                            'lat': coordinates['lat'],
                            'lon': coordinates['lon'],
                            'name': 'Your Location 📍'
                        }])
                        
                        all_map_data = pd.concat([map_data, user_marker], ignore_index=True)
                        st.map(all_map_data)
                        
                        # Display collection point details in a table
                        st.subheader("Collection point details")
                        
                        # Create a table to display the details
                        table_data = []
                        for point in points:
                            # Convert API waste types to UI format
                            ui_waste_types_list = [convert_api_to_ui_waste_type(wt) for wt in point["waste_types"]]
                            
                            table_data.append({
                                "Name": point["name"],
                                "Address": point["address"],
                                "Distance (km)": point["distance"],
                                "Accepted waste types": ", ".join(ui_waste_types_list),
                                "Hours": point["hours"] if point["hours"] else "Not specified"
                            })
                        
                        # Sort by distance
                        df = pd.DataFrame(table_data)
                        df = df.sort_values("Distance (km)")
                        st.table(df)
                        
                        # Search for upcoming collection dates
                        st.subheader("Upcoming collection dates")
                        with st.spinner("Searching for collection dates..."):
                            collection_dates = get_collection_dates(api_waste_type, user_location)
                            
                            if collection_dates:
                                # Display collection dates in a table
                                date_data = []
                                for date_info in collection_dates[:5]:  # Limit to 5 dates
                                    date_obj = datetime.strptime(date_info["date"], "%Y-%m-%d")
                                    formatted_date = date_obj.strftime("%d/%m/%Y")
                                    
                                    date_data.append({
                                        "Date": formatted_date,
                                        "Time": date_info["time"],
                                        "Area": date_info["area"],
                                        "Title": date_info["title"]
                                    })
                                
                                st.table(pd.DataFrame(date_data))
                                
                                # Link to PDF if available
                                if collection_dates[0].get("pdf"):
                                    st.markdown(f"[Download the complete collection calendar (PDF)]({collection_dates[0]['pdf']})")
                            else:
                                st.info("No collection dates were found for this type of waste at your address.")
                    else:
                        st.warning(f"No collection points for {waste_type} were found within 10 km of your location. Try searching for a different waste type or check if you entered the correct address.")
                else:
                    st.error("Unable to locate the provided address. Please make sure to enter a complete address in St. Gallen, Switzerland.")
        else:
            st.error("Please enter your full address to search for collection points")

with tab2:
    st.header("Identify your waste")

    # Load text model
    @st.cache_resource
    def load_text_model():
        try:
            with open('waste_classifier.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('waste_vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            with open('waste_encoder.pkl', 'rb') as f:
                encoder = pickle.load(f)
            return model, vectorizer, encoder
        except FileNotFoundError:
            return None, None, None

    # Load fine-tuned image model
    @st.cache_resource
    def load_image_model():
        try:
            from tensorflow.keras.models import load_model
            return load_model("waste_image_classifier.h5")
        except Exception as e:
            logger.warning(f"Could not load image model: {e}")
            return None

    def predict_from_text(description, model, vectorizer, encoder):
        description = description.lower()
        X_new = vectorizer.transform([description])
        prediction = model.predict(X_new)[0]
        probabilities = model.predict_proba(X_new)[0]
        confidence = probabilities[prediction]
        category = encoder.inverse_transform([prediction])[0]
        return category, confidence

    def predict_from_image(img, model, class_names):
        try:
            from tensorflow.keras.preprocessing import image as keras_image

            img = img.resize((224, 224))
            img_array = keras_image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions)]
            confidence = np.max(predictions)

            return predicted_class, confidence
        except Exception as e:
            logger.error(f"Error in image prediction: {e}")
            st.error(f"Error analyzing image: {e}")
            return None, 0.0

    # Load models
    text_model, text_vectorizer, text_encoder = load_text_model()
    image_model = load_image_model()
    image_class_names = [
        "Household waste 🗑", "Paper 📄", "Cardboard 📦", "Glass 🍾", "Green waste 🌿", 
        "Cans 🥫", "Aluminium 🧴", "Foam packaging ☁", "Metal 🪙", "Textiles 👕", 
        "Oil 🛢", "Hazardous waste ⚠"
    ]

    # Input from user
    waste_description = st.text_area("Describe your waste (material, size, usage, etc.)")
    uploaded_file = st.file_uploader("Or upload a photo of your waste", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", width=300)

    if st.button("Identify"):
        if waste_description or uploaded_file:
            with st.spinner("Analyzing your waste..."):

                if waste_description and text_model:
                    category, confidence = predict_from_text(waste_description, text_model, text_vectorizer, text_encoder)
                    st.success(f"Text analysis result: {category} (confidence: {confidence:.2%})")

                elif uploaded_file and image_model:
                    category, confidence = predict_from_image(image, image_model, image_class_names)
                    if category:
                        st.success(f"Image analysis result: {category} (confidence: {confidence:.2%})")

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
        st.error("❌ Text Classification Model: Not found")
    
    if os.path.exists("waste_image_classifier.h5"):
        if check_tensorflow_available():
            st.success("✅ Image Classification Model: Available")
        else:
            st.warning("⚠️ Image Classification Model: File exists but TensorFlow not installed")
    else:
        st.error("❌ Image Classification Model: Not found")
    
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
    
    # Dark/light mode
    if st.checkbox("Dark mode", value=False):
        st.markdown("""
        <style>
        .stApp {
            background-color: #121212;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Application language (simulated)
    st.selectbox("Language", ["English", "Français", "Deutsch", "Italiano"])
    
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
    
    # St. Gallen collection points map
    st.subheader("General map")
    
    # Display a small map centered on St. Gallen
    st_gallen_map = pd.DataFrame({
        'lat': [47.4245],
        'lon': [9.3767]
    })
    st.map(st_gallen_map)
    
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