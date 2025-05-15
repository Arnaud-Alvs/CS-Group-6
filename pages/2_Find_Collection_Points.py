# This is the page displayed on the app with the information/ function to find collection points
# It uses the information from location_api 

"""
Find Collection Points page for the WasteWise application.
"""

# Import necessary libraries to run the program
import streamlit as st
import sys
import os
import folium 
from streamlit_folium import st_folium


# Set Streamlit page settings (title, icon, layout, sidebar state)
st.set_page_config(
    page_title="WasteWise - Find Collection Points",
    page_icon="🚮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit navigation elements that are not necessary
hide_streamlit_style = """
<style>
[data-testid="stSidebarNavItems"] {
    display: none !important;
}
button[kind="header"] {
    display: none !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 1rem !important;
}
.sidebar-content .sidebar-collapse-control {
    display: none !important;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Add the parent directory to the path to access app.py functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import all the helper functions from our location_api module
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
except ImportError as e:
    st.error(f"Failed to import required functions: {str(e)}")
    st.stop()

# Store results in session so we can reuse or redisplay them
if 'waste_info_results' not in st.session_state:
    st.session_state.waste_info_results = None
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'user_address' not in st.session_state:
    st.session_state.user_address = ""
if 'selected_waste_type' not in st.session_state:
    st.session_state.selected_waste_type = ""

# Display the main title of the page
st.title("🚮 Find Collection Information")

# Determine if user came from image/text identification
coming_from_identification = (
    st.session_state.show_results and 
    st.session_state.waste_info_results is not None and
    hasattr(st.session_state, 'identified_waste_type') and
    st.session_state.identified_waste_type is not None
)
# Show welcome instructions if no results have been loaded yet
if not st.session_state.show_results:
    st.markdown(
        """
        Welcome to WasteWise! Enter the type of waste you want to dispose of
        and your address in St. Gallen to find nearby collection points and
        upcoming collection dates.
        """
    )
# Show the input form if results don't exist or user wants to search again
if not st.session_state.show_results or st.checkbox("Search for a different waste type or address"):
    # Get available waste types from the API
    available_waste_types_german = get_available_waste_types()
    available_waste_types_english = [translate_waste_type(wt) for wt in available_waste_types_german]
    waste_type_mapping = dict(zip(available_waste_types_english, available_waste_types_german))

    # Form where the user chooses the waste type and enters their address (input from the user)
    with st.form(key="waste_search_form"):
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
            
                # Get waste disposal information
                waste_info = handle_waste_disposal(user_address, selected_waste_type_german)
            
                # Store results in session state
                st.session_state.waste_info_results = waste_info
                st.session_state.show_results = True
                st.rerun()

# Display results section - only if we have data
if 'show_results' in st.session_state and st.session_state.show_results:
    # Clear separation from the input form
    st.markdown("---")
    
    # Results container
    results_container = st.container()
    
    with results_container:
        waste_info = st.session_state.waste_info_results
        selected_waste_type_english = st.session_state.selected_waste_type
        user_address = st.session_state.user_address
        
        st.subheader("Search Results")
        st.markdown(waste_info["message"])
        
        # Display collection points and map if available
        if waste_info["has_disposal_locations"]:
            st.markdown(f"### Nearest Collection Points for {selected_waste_type_english}")
            
            # Get user coordinates for the map
            user_coords = get_coordinates(user_address)
            
            if user_coords:
                map_col, info_col = st.columns([3, 2])
                
                with map_col:
                    st.caption("Hover over markers for info, click for details.")
                    
                    # Create interactive map
                    interactive_map = create_interactive_map(user_coords, waste_info["collection_points"])
                    
                    # Display map
                    st_folium(
                        interactive_map, 
                        width=None,
                        height=400,
                        returned_objects=[],
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
            next_collection = waste_info["next_collection_date"]
            
            # Format collection date
            collection_date = next_collection['date'].strftime('%A, %B %d, %Y')
            collection_time = next_collection.get('time', '')
            
            # Display in a highlighted box
            date_html = f"""
            <div style="background-color: #d4edda; border-radius: 5px; padding: 15px; margin-bottom: 15px;">
                <h3 style="color: #155724; margin-top: 0;">Next Collection: {collection_date}</h3>
                {f"<p style='margin-bottom: 0;'><strong>Time:</strong> {collection_time}</p>" if collection_time and collection_time != "N/A" else ""}
                {f"<p style='margin-bottom: 0;'><strong>Details:</strong> {next_collection['description']}</p>" if next_collection['description'] and next_collection['description'] != "Collection" else ""}
                {f"<p style='margin-bottom: 0;'><strong>Area:</strong> {next_collection['area']}</p>" if next_collection['area'] and next_collection['area'] != "N/A" else ""}
            </div>
            """
            st.markdown(date_html, unsafe_allow_html=True)

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

# This file was created with the help of Claude AI to help with correcting mistakes - Arnaud Alves
# The documentation (comments) were done with the help of ChatGPT - Noah Pittet
