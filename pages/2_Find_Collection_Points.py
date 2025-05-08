import streamlit as st

# Page configuration
st.set_page_config(
    page_title="WasteWise - Find Collection Points",
    page_icon="🚮",
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
import random
import sys
import os
import folium 
from streamlit_folium import st_folium

# Add the parent directory to the path to access app.py functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required functions from the root app module
try:
    from app import (
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
except ImportError as e:
    st.error(f"Failed to import required functions: {str(e)}")
    st.stop()

# Initialize session state if not already done
if 'waste_info_results' not in st.session_state:
    st.session_state.waste_info_results = None
    st.session_state.show_results = False

# Page header
st.title("🚮 Find Collection Information")

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
# Useful links
st.subheader("Useful links")
st.markdown("[Complete recycling guide](https://www.stadt.sg.ch/home/umwelt-energie/entsorgung.html)")
st.markdown("[Reducing waste in everyday life](https://www.bafu.admin.ch/bafu/en/home/topics/waste/guide-to-waste-a-z/avoiding-waste.html)")
st.markdown("[Waste legislation in Switzerland](https://www.bafu.admin.ch/bafu/en/home/topics/waste/legal-basis.html)")
st.markdown("[Official St. Gallen city website](https://www.stadt.sg.ch/)")

# Set up sidebar
with st.sidebar:
    st.title("WasteWise")
    st.markdown("Your smart recycling assistant")
    
    # Navigation
    st.markdown("## Navigation")
    st.page_link("1_Home.py", label="Home", icon="🏠")
    st.page_link("2_Find_Collection_Points.py", label="Find Collection Points", icon="🚮")
    st.page_link("3_Identify_Waste.py", label="Identify Waste", icon="🔍")
    st.page_link("4_About.py", label="About", icon="ℹ️")
    
    # Useful links
    st.markdown("## Useful Links")
    st.markdown("[Complete recycling guide](https://www.stadt.sg.ch/home/umwelt-energie/entsorgung.html)")
    st.markdown("[Reducing waste in everyday life](https://www.bafu.admin.ch/bafu/en/home/topics/waste/guide-to-waste-a-z/avoiding-waste.html)")
    st.markdown("[Waste legislation in Switzerland](https://www.bafu.admin.ch/bafu/en/home/topics/waste/legal-basis.html)")
    st.markdown("[Official St. Gallen city website](https://www.stadt.sg.ch/)")

# Footer
st.markdown("---")
st.markdown("© 2025 WasteWise - University Project | [Contact](mailto:contact@wastewise.example.com) | [Legal notice](https://example.com)")
