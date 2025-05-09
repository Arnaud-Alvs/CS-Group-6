import streamlit as st

st.set_page_config(
    page_title="WasteWise - About",
    page_icon="ℹ️",
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
import requests
import sys
import os
import time
from PIL import Image 

# Add the parent directory to the path to access app.py functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import any functions we might need
try:
    from app import (
        check_ml_models_available, 
        check_tensorflow_available
    )
except ImportError as e:
    st.error(f"Failed to import required functions: {str(e)}")
    
# Page header
st.title("ℹ️ About WasteWise")

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
""")

# Team section
st.markdown("## Our Team")

# Create a nicer team layout with columns
col1, col2, col3, col4, col5= st.columns(5)

image_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "photos")

with col1:
    st.markdown("### Project Lead")
    photo1 = Image.open("photos/Andreas-Jonas-Lucchini.jpg")
    st.image(photo1, width=150)
    st.markdown("**Andreas l'italien**")
    st.markdown("Porject Management")
    
with col2:
    st.markdown("### Backend Developer")
    photo2 = Image.open("photos/Andreas-Jonas-Lucchini.jpg")
    st.image(photo1, width=150)
    st.markdown("**Arnaud Alves**")
    st.markdown("API Integration & Machine Learning")
    
with col3:
    st.markdown("### Frontend Developer")
    photo3 = Image.open("photos/arnaud_butty.jpg")
    st.image(photo3, width=150)
    st.markdown("**Arnaud Butty**")
    st.markdown("API Integration & Machine Learning")

with col4:
    st.markdown("### Frontend Developer")
    photo4 = Image.open("photos/noah.jpg")
    st.image(photo4, width=150)
    st.markdown("**Noah Pittet**")
    st.markdown("UI/UX Design")

with col5:
    st.markdown("### Video creation")
    photo5 = Image.open("photos/seb.jpg")
    st.image(photo5, width=150)
    st.markdown("**Sebastien Carriage**")
    st.markdown("Creation of the presentation video")

# Project timeline
st.markdown("## Project Timeline")

# Create a more visual timeline
timeline_data = [
    {"date": "January 2025", "milestone": "Project Inception", "description": "Initial concept development and team formation"},
    {"date": "February 2025", "milestone": "Data Collection", "description": "API integration and waste dataset compilation"},
    {"date": "March 2025", "milestone": "Model Development", "description": "Training of waste classification models"},
    {"date": "April 2025", "milestone": "Beta Release", "description": "Internal testing and bug fixing"},
    {"date": "May 2025", "milestone": "Public Launch", "description": "Official release of WasteWise application"}
]

# Display timeline as a table with custom styling
st.markdown("""
<style>
    .timeline-table {
        width: 100%;
        border-collapse: collapse;
    }
    .timeline-table td {
        padding: 15px;
        border-bottom: 1px solid #f0f2f6;
    }
    .milestone {
        font-weight: bold;
        color: #4CAF50;
    }
    .date {
        color: #777;
        width: 130px;
    }
</style>
<table class="timeline-table">
""", unsafe_allow_html=True)

for item in timeline_data:
    st.markdown(f"""
    <tr>
        <td class="date">{item["date"]}</td>
        <td class="milestone">{item["milestone"]}</td>
        <td>{item["description"]}</td>
    </tr>
    """, unsafe_allow_html=True)

st.markdown("</table>", unsafe_allow_html=True)

# User feedback section
st.markdown("## Your feedback matters")
st.write("Help us improve WasteWise by sharing your suggestions:")

with st.form("feedback_form"):
    feedback_text = st.text_area("Your comments and suggestions")
    user_email = st.text_input("Your email (optional)")
    submit_button = st.form_submit_button("Send my feedback")
    
    if submit_button:
        if feedback_text:
            # Simulate sending feedback
            with st.spinner("Sending feedback..."):
                time.sleep(1)  # Simulate network delay
                st.success("Thank you for your feedback! We have received it.")
                # In a real project, you would record this feedback in a database
        else:
            st.error("Please enter a comment before sending.")

# Privacy policy / Terms of service
st.markdown("## Privacy & Terms")
with st.expander("Privacy Policy"):
    st.markdown("""
    ### Privacy Policy
    
    **WasteWise** values your privacy and is committed to protecting your personal data.
    
    - We collect minimal data required to provide our services
    - Your address information is only used to find nearby collection points
    - We do not share your data with third parties
    - Images uploaded for waste identification are not stored permanently
    
    For more information, please contact our data protection officer at privacy@wastewise.example.com
    """)
    
with st.expander("Terms of Service"):
    st.markdown("""
    ### Terms of Service
    
    By using the **WasteWise** application, you agree to the following terms:
    
    - This is an educational project and should not replace official waste management guidelines
    - While we strive for accuracy, we cannot guarantee 100% correctness of waste identification
    - Users should always verify disposal information with local authorities
    - The application is provided "as is" without warranties of any kind
    
    For any questions regarding these terms, please contact legal@wastewise.example.com
    """)

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
