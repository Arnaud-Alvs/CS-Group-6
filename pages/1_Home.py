# Import necessary libraries
import streamlit as st
import random
from PIL import Image
import os

# Set up page-wide configuration (must be done first before anything else appears)
st.set_page_config(
    page_title="WasteWise - Home",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide default Streamlit elements that we don’t want (like built-in navigation)
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
# Apply the custom style
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Main content

# Display the title at the top of the homepage
st.title("♻️ WasteWise - Your smart recycling assistant")


# Describe the main features with headings and buttons
st.markdown("## What can WasteWise do for you?")

# Divide the space into two columns
col1, col2 = st.columns(2)

# Left column — Finding collection points
with col1:
    st.markdown("### 🚮 Find Collection Points")
    st.markdown("""
    Simply tell us what waste you want to dispose of and your address in St. Gallen.
    We'll find the nearest collection points and upcoming collection dates for you.
    """)
    st.page_link("pages/2_Find_Collection_Points.py", label="Find Disposal Options", icon="🔍")

# Right column — Identifying waste
with col2:
    st.markdown("### 🔍 Identify Your Waste")
    st.markdown("""
    Not sure what type of waste you have? Upload a photo or describe it,
    and our AI will help you identify it and provide proper disposal instructions.
    """)
    st.page_link("pages/3_Identify_Waste.py", label="Identify Your Waste", icon="📸")

# Tips of the day section
st.markdown("---")
st.subheader("💡 Tip of the Day")
# List of random environmental tips
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
# Randomly show one tip each time the page is refreshed
st.info(random.choice(tips_of_the_day))

# Environmental impact section
st.markdown("## Environmental Impact")

# Split the screen into 3 columns to show key impact metrics
impact_col1, impact_col2, impact_col3 = st.columns(3)

with impact_col1:
    st.metric(label="Waste Correctly Sorted", value="12,543 kg", delta="1,243 kg this month")

with impact_col2:
    st.metric(label="CO₂ Emissions Saved", value="2,432 kg", delta="347 kg this month")

with impact_col3:
    st.metric(label="Active Users", value="853", delta="57 new")


# SIDEBAR SETUP
with st.sidebar:
    st.title("WasteWise")
    st.markdown("Your smart recycling assistant")
    
    # Main navigation links
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
