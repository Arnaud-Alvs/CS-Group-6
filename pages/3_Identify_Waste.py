# imports streamlit 
import streamlit as st

# sets up the page configuration, with a title, icon, and layout
st.set_page_config(
    # We define the title that will appear in the browser tab, set the icon, define the layout to use the full width of the screen
    page_title="WasteWise - Identify Waste",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide ALL built-in Streamlit navigation elements
# We create a custom CSS style to hide the default Streamlit sidebar navigation

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

# We import all the necessary libraries to make our waste identification system work properly
# pandas is used for data manipulation, numpy for mathematical operations
# requests allows us to make API calls to external services if needed
import pandas as pd
import numpy as np
from PIL import Image
import requests
import sys
import os

# Add the parent directory to the path to access app.py functions
# This is necessary because we need to use functions that are defined in our main app.py file
# This approach allows us to maintain a modular code structure while avoiding code duplication
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We try to import all the necessary functions from app.py that we need for waste identification
# These functions include model loading, prediction functions, and utilities to handle waste disposal information
# If the import fails, we show an error message and stop the application to prevent further issues
try:
    from app import (
        check_ml_models_available,
        check_tensorflow_available,
        load_text_model,
        load_image_model,
        predict_from_text,
        predict_from_image,
        convert_waste_type_to_api,
        handle_waste_disposal,
        translate_waste_type,
        get_available_waste_types,
    )
except ImportError as e:
    st.error(f"Failed to import required functions: {str(e)}")
    st.stop()

# We define the waste categories that our image recognition model can identify
# This list corresponds to the classes that our ML model was trained on
IMAGE_CLASS_NAMES = [
    "Aluminium 🧴", "Cans 🥫", "Cardboard 📦", "Foam packaging ☁", "Glass 🍾", "Green waste 🌿",
    "Hazardous waste ⚠", "Household waste 🗑", "Metal 🪙", "Oil 🛢", "Paper 📄", "Plastic", "Textiles 👕"
]

# Initialize session state for waste identification
# Session state allows us to maintain data between reruns of the Streamlit app
# This ensures that the user's identification results persist as they navigate through the app
if 'identified_waste_type' not in st.session_state:
    st.session_state.identified_waste_type = None
    st.session_state.waste_confidence = None
    st.session_state.search_for_collection = False

# Page header
# We create a title for our page with a magnifying glass emoji to represent the search functionality
st.title("🔍 Identify Your Waste")
st.markdown("""
Use our AI-powered waste identification system to determine what type of waste you have
and learn how to dispose of it properly. You can either describe your waste or upload a photo.
""")

# Check if ML models are available and load them once
# We load our text and image models that will identify waste types based on descriptions or photos
text_model, text_vectorizer, text_encoder = load_text_model()
image_model = load_image_model()

# We create two columns to display the status of our ML models
# If a model isn't available, we inform users that we'll use a fallback method
# This lets users know which identification methods are available
tab1, tab2 = st.tabs(["Describe your waste", "Upload a photo"])

# First tab: Identify waste through text description
# This tab contains a text area where users can describe their waste
# The system will analyze this description to identify the waste type 
with tab1:
    st.markdown("### Describe your waste")
    waste_description = st.text_area(
        "Describe the material, size, previous use, etc.", 
        placeholder="Example: A clear glass bottle that contained olive oil",
        height=100
    )
    
    # When the "Identify" button is clicked, we process the text description
    # We check if the description is empty and show a warning if it is
    if st.button("Identify from Description", key="identify_text"):
        if not waste_description:
            st.warning("Please enter a description first.")
        else:
            with st.spinner("Analyzing your waste description..."):
                category, confidence = predict_from_text(
                    waste_description, 
                    model=text_model, 
                    vectorizer=text_vectorizer, 
                    encoder=text_encoder
                )
                if category:
                    st.session_state.identified_waste_type = category
                    st.session_state.waste_confidence = confidence
                    st.success(f"Text analysis result: {category} (confidence: {confidence:.2%})")
                    
                    if category == "Unknown 🚫":
                        st.warning("⚠️ This item does not match any known waste category. Please try describing it differently.")
                    else:
                        st.session_state.search_for_collection = True
                        st.rerun()

# Second tab: Identify waste through image upload
# This tab allows users to upload an image of their waste
# The system will analyze this image to identify the waste type

with tab2:
    st.markdown("### Upload a photo of your waste")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    # Display uploaded image and process it when the Identify button is clicked
    # We show the uploaded image to the user and then analyze it using our image model
    # The identified waste type and confidence level are stored in the session state
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", width=300)
            
            if st.button("Identify from Image", key="identify_image"):
                with st.spinner("Analyzing your waste image..."):
                    # Call predict_from_image without class_names parameter
                    # The function will use MODEL_CLASS_NAMES internally
                    category, confidence = predict_from_image(
                        image, 
                        model=image_model
                    )
                    if category:
                        st.session_state.identified_waste_type = category
                        st.session_state.waste_confidence = confidence
                        st.success(f"Image analysis result: {category} (confidence: {confidence:.2%})")
                        
                        if category == "Unknown 🚫":
                            st.warning("⚠️ This item could not be identified. Try uploading a clearer image or use the description method.")
                        else:
                            st.session_state.search_for_collection = True
                            st.rerun()
        except Exception as e:
            st.error(f"Error processing image: {e}")

# Display identification results and sorting advice
if st.session_state.identified_waste_type:
    st.markdown("---")
    st.header(f"Results: {st.session_state.identified_waste_type}")
    
    # Format confidence as percentage and display as a progress bar
    confidence_pct = st.session_state.waste_confidence * 100
    st.progress(min(confidence_pct/100, 1.0), text=f"Confidence: {confidence_pct:.1f}%")
    
    # shows sorting advice based on the identified waste type
    if st.session_state.identified_waste_type != "Unknown 🚫":
        st.subheader("Waste sorting advice")
        
        # extracts the base category without emoji
        category = st.session_state.identified_waste_type
        base_category = category.split(" ")[0] if " " in category else category
        
        # defines sorting advice for different waste types
        # For each waste type, we provide information on the appropriate bin and disposal tips

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
            },
            "Plastic": { 
                "bin": "Plastic recycling or general waste (check local rules)",
                "tips": [ 
                    "Rinse bottles and containers",
                    "Remove caps and labels if required",
                    "Check for recyclable plastic symbols",
                    "Do not mix with other materials"
                ]
            }
        }
        
        # Find matching advice for the identified waste type
        # We search through our sorting advice dictionary to find the appropriate guidance
        # If no match is found, we provide a default message
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

# offers the user to search for collection points if a valid waste type was identified
if st.session_state.identified_waste_type != "Unknown 🚫" and st.session_state.search_for_collection:
    st.markdown("---")
    st.subheader("Find collection points")
    
    # Convert identified waste type to API format
    # Our API uses a different format for waste types, so we need to convert
    api_waste_type = convert_waste_type_to_api(st.session_state.identified_waste_type)

    # Create a form for users to enter their address
    with st.form(key="identified_waste_form"):
        user_address = st.text_input(
            "Enter your address in St. Gallen to find nearby collection points:",
            placeholder="e.g., Musterstrasse 1",
            help="Enter your address, it must include a street name and number."
        )
        
        submit_button = st.form_submit_button("Find collection points")

    # Handle the form submission
    # When the user submits the form, we process their address and waste type
    if submit_button:
        if not user_address:
            st.warning("Please enter your address.")
        else:
            with st.spinner(f"Finding collection options..."):
                # Convert identified waste type to API format
                ui_waste_type = st.session_state.identified_waste_type
                api_waste_type = convert_waste_type_to_api(ui_waste_type)
                
                # Call the handle_waste_disposal function with the API-formatted waste type
                # This function communicates with our backend to find collection points
                waste_info = handle_waste_disposal(user_address, api_waste_type)
                
                # Store results in session state for Page 2 to use
                # We save the results so they can be displayed on the collection points page
                st.session_state.waste_info_results = waste_info
                st.session_state.selected_waste_type = ui_waste_type
                st.session_state.user_address = user_address
                st.session_state.show_results = True
                
                # Only show the button to view results
                st.markdown("""
                <div style="text-align: center; margin-top: 20px; margin-bottom: 20px;">
                    <h3>Click below to view detailed collection options.</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Use a direct link to Page 2, centered and with a bigger button
                # This provides a clear call to action for users to view their results
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.page_link(
                        "pages/2_Find_Collection_Points.py",
                        label="➡️ View Collection options",
                        use_container_width=True
                    )

# shows the reset button at the bottom of the page
if st.session_state.identified_waste_type:
    if st.button("Start Over", key="start_over"):
        # Reset all session state variables
        st.session_state.identified_waste_type = None
        st.session_state.waste_confidence = None
        st.session_state.search_for_collection = False
        st.rerun()

# set up sidebar
with st.sidebar:
    st.title("WasteWise")
    st.markdown("Your smart recycling assistant")
    
    # displays the navigation menu
    st.markdown("## Navigation")
    st.page_link("pages/1_Home.py", label="Home", icon="🏠")
    st.page_link("pages/2_Find_Collection_Points.py", label="Find Collection Points", icon="🚮")
    st.page_link("pages/3_Identify_Waste.py", label="Identify Waste", icon="🔍")
    st.page_link("pages/4_About.py", label="About", icon="ℹ️")
    
    # displays useful links
    st.markdown("## Useful Links")
    st.markdown("[Complete recycling guide](https://www.stadt.sg.ch/home/umwelt-energie/entsorgung.html)")
    st.markdown("[Reducing waste in everyday life](https://www.bafu.admin.ch/bafu/en/home/topics/waste/guide-to-waste-a-z/avoiding-waste.html)")
    st.markdown("[Waste legislation in Switzerland](https://www.bafu.admin.ch/bafu/en/home/topics/waste/legal-basis.html)")
    st.markdown("[Official St. Gallen city website](https://www.stadt.sg.ch/)")

# Footer
st.markdown("---")
st.markdown("© 2025 WasteWise - University Project")

# Th
