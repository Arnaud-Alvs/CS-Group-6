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

# Team section
st.markdown("## Our Team")

# Create a nicer team layout with columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Project Lead")
    st.image("https://via.placeholder.com/150", width=150)
    st.markdown("**Marco Rossi**")
    st.markdown("Data Science & Project Management")
    
with col2:
    st.markdown("### Backend Developer")
    st.image("https://via.placeholder.com/150", width=150)
    st.markdown("**Sophie Müller**")
    st.markdown("API Integration & Machine Learning")
    
with col3:
    st.markdown("### Frontend Developer")
    st.image("https://via.placeholder.com/150", width=150)
    st.markdown("**Alex Chen**")
    st.markdown("UI/UX Design & Streamlit Development")

# Display fictional statistics on application usage
st.markdown("## WasteWise Impact")

# Use columns to display statistics side by side
stat_col1, stat_col2, stat_col3 = st.columns(3)

with stat_col1:
    st.metric(label="Waste identified", value="12,543", delta="1,243 this week")

with stat_col2:
    st.metric(label="Collection points", value="3,879", delta="152 new")

with stat_col3:
    st.metric(label="Active users", value="853", delta="57 new")

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
