import streamlit as st

# Page configuration must be the FIRST Streamlit command
st.set_page_config(
    page_title="WasteWise - Identify Waste",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide ALL built-in Streamlit navigation elements
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

import pandas as pd
import numpy as np
from PIL import Image
import requests
import sys
import os

# Add the parent directory to the path to access app.py functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required functions from the root app module
try:
    from app import (
        check_ml_models_available,
        check_tensorflow_available,
        load_text_model,
        load_image_model,
        predict_from_text,
        predict_from_image,
        rule_based_prediction,
        simple_image_prediction,
        IMAGE_CLASS_NAMES,
        convert_waste_type_to_api,
        handle_waste_disposal,
        translate_waste_type,
        get_available_waste_types,
    )
except ImportError as e:
    st.error(f"Failed to import required functions: {str(e)}")
    st.stop()


# Initialize session state for waste identification
if 'identified_waste_type' not in st.session_state:
    st.session_state.identified_waste_type = None
    st.session_state.waste_confidence = None
    st.session_state.search_for_collection = False

# Page header
st.title("🔍 Identify Your Waste")
st.markdown("""
Use our AI-powered waste identification system to determine what type of waste you have
and learn how to dispose of it properly. You can either describe your waste or upload a photo.
""")

# Check if ML models are available and load them once
text_model, text_vectorizer, text_encoder = load_text_model()
image_model = load_image_model()

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

# Create tabs for different input methods
tab1, tab2 = st.tabs(["Describe your waste", "Upload a photo"])

with tab1:
    st.markdown("### Describe your waste")
    waste_description = st.text_area(
        "Describe the material, size, previous use, etc.", 
        placeholder="Example: A clear glass bottle that contained olive oil",
        height=100
    )
    
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

with tab2:
    st.markdown("### Upload a photo of your waste")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    # Display uploaded image
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", width=300)
            
            if st.button("Identify from Image", key="identify_image"):
                with st.spinner("Analyzing your waste image..."):
                    category, confidence = predict_from_image(
                        image, 
                        model=image_model, 
                        class_names=IMAGE_CLASS_NAMES
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
    
    # Format confidence as percentage
    confidence_pct = st.session_state.waste_confidence * 100
    st.progress(min(confidence_pct/100, 1.0), text=f"Confidence: {confidence_pct:.1f}%")
    
    # Show sorting advice for the predicted category
    if st.session_state.identified_waste_type != "Unknown 🚫":
        st.subheader("Waste sorting advice")
        
        # Extract the base category without emoji
        category = st.session_state.identified_waste_type
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
            },
            "Plastic": { 
                "bin": "Plastic recycling or general waste (check local rules)",
                "tips": [ 
                    "Rinse bottles and containers"
                    "Remove caps and labels if required",
                    "Check for recyclable plastic symbols",
                    "Do not mix with other materials"
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

# Offer to search for collection points if a valid waste type was identified
if st.session_state.identified_waste_type != "Unknown 🚫" and st.session_state.search_for_collection:
    st.markdown("---")
    st.subheader("Find collection points")
    
    # Convert identified waste type to API format
    api_waste_type = convert_waste_type_to_api(st.session_state.identified_waste_type)
    
    with st.form(key="identified_waste_form"):
        user_address = st.text_input(
            "Enter your address in St. Gallen to find nearby collection points:",
            placeholder="e.g., Musterstrasse 1",
            help="Enter your address, it must include a street name and number."
        )
        
        submit_button = st.form_submit_button("Find collection points")
    
if submit_button:
    if not user_address:
        st.warning("Please enter your address.")
    else:
        with st.spinner(f"Searching for disposal options for {st.session_state.identified_waste_type}..."):
            # Convert identified waste type to API format with better debugging
            ui_waste_type = st.session_state.identified_waste_type
            api_waste_type = convert_waste_type_to_api(ui_waste_type)
            
            # Debug the conversion
            st.write(f"Converting UI waste type: '{ui_waste_type}' to API format: '{api_waste_type}'")
            
            # Call the handle_waste_disposal function with the API-formatted waste type
            waste_info = handle_waste_disposal(user_address, api_waste_type)
            
            # Debug what the API returned
            st.write(f"API response has_disposal_locations: {waste_info['has_disposal_locations']}")
            st.write(f"API found: {len(waste_info['collection_points'])} collection points")
            
            # Store results in session state for Page 2 to use
            st.session_state.waste_info_results = waste_info
            st.session_state.selected_waste_type = ui_waste_type  # Use the UI waste type for display
            st.session_state.user_address = user_address
            st.session_state.show_results = True  # Flag to show results in Page 2
            
            # Show the actual waste_info message 
            st.markdown("### Results:")
            st.markdown(waste_info["message"])
            
            if waste_info["has_disposal_locations"]:
                points_list = waste_info['collection_points']
                st.markdown(f"**Found {len(points_list)} collection points**")
                
                # Show the first 3 collection points as a preview
                if len(points_list) > 0:
                    st.markdown("### Nearest collection points:")
                    for i, point in enumerate(points_list[:3]):
                        st.markdown(f"**{i+1}. {point['name']}** ({point['distance']:.2f} km)")
                        waste_types = [translate_waste_type(wt) for wt in point['waste_types']]
                        st.markdown(f"Accepts: {', '.join(waste_types)}")
            
            if waste_info["has_scheduled_collection"]:
                next_date = waste_info["next_collection_date"]["date"].strftime('%A, %B %d, %Y')
                st.markdown(f"**Next collection date: {next_date}**")
            
            # Create a container for the button to navigate to Page 2
            button_container = st.container()
            
            with button_container:
                st.markdown("### View detailed results with interactive map")
                
                # Use a direct link instead of a button with switch_page
                st.page_link(
                    "pages/2_Find_Collection_Points.py",
                    label="Go to Collection Points Page",
                    icon="🗺️"
                )
# Add this at the bottom of the find collection points section
# but before the "Reset button" section in pages/3_Identify_Waste.py

# Add a direct test section for debugging
st.markdown("---")
st.subheader("Debug: Test API Directly")
st.markdown("If you're having trouble finding collection points, you can test the API directly here.")

test_col1, test_col2 = st.columns(2)

with test_col1:
    # List all available waste types from the API
    st.markdown("**Available waste types (API format):**")
    api_waste_types = get_available_waste_types()
    for wt in api_waste_types:
        st.write(f"- {wt} → {translate_waste_type(wt)}")

with test_col2:
    # Direct test form
    st.markdown("**Test direct API call:**")
    
    with st.form(key="direct_api_test"):
        test_waste_type = st.selectbox(
            "Select API waste type:",
            options=api_waste_types
        )
        
        test_address = st.text_input(
            "Test address:",
            placeholder="e.g., Musterstrasse 1, St. Gallen"
        )
        
        test_button = st.form_submit_button("Test API call")
    
    if test_button and test_waste_type and test_address:
        with st.spinner("Testing API..."):
            # Call API directly with the selected waste type
            waste_info = handle_waste_disposal(test_address, test_waste_type)
            
            st.write(f"API call result: {len(waste_info['collection_points'])} collection points found")
            
            if len(waste_info['collection_points']) > 0:
                st.success("Success! Collection points found.")
                st.markdown("First point: " + waste_info['collection_points'][0]['name'])
            else:
                st.error("No collection points found with direct API call.")
                st.markdown("This suggests an issue with the API connection or the selected waste type/address.")
# Reset button at the bottom of the page
if st.session_state.identified_waste_type:
    if st.button("Start Over", key="start_over"):
        # Reset all session state variables
        st.session_state.identified_waste_type = None
        st.session_state.waste_confidence = None
        st.session_state.search_for_collection = False
        st.rerun()

# Set up sidebar
with st.sidebar:
    st.title("WasteWise")
    st.markdown("Your smart recycling assistant")
    
    # Navigation
    st.markdown("## Navigation")
    st.page_link("pages/1_Home.py", label="Home", icon="🏠")
    st.page_link("pages/2_Find_Collection_Points.py", label="Find Collection Points", icon="🚮")
    st.page_link("pages/3_Identify_Waste.py", label="Identify Waste", icon="🔍")
    st.page_link("pages/4_About.py", label="About", icon="ℹ️")
    
    # Useful links
    st.markdown("## Useful Links")
    st.markdown("[Complete recycling guide](https://www.stadt.sg.ch/home/umwelt-energie/entsorgung.html)")
    st.markdown("[Reducing waste in everyday life](https://www.bafu.admin.ch/bafu/en/home/topics/waste/guide-to-waste-a-z/avoiding-waste.html)")
    st.markdown("[Waste legislation in Switzerland](https://www.bafu.admin.ch/bafu/en/home/topics/waste/legal-basis.html)")
    st.markdown("[Official St. Gallen city website](https://www.stadt.sg.ch/)")

# Footer
st.markdown("---")
st.markdown("© 2025 WasteWise - University Project | [Contact](mailto:contact@wastewise.example.com) | [Legal notice](https://example.com)")
