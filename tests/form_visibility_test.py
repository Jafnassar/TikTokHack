"""
Form Visibility Test for Dark Theme
"""

import streamlit as st

st.set_page_config(page_title="Form Visibility Test", layout="wide")

# Apply comprehensive CSS fixes for form visibility
st.markdown("""
<style>
    /* Form labels - light text for dark backgrounds */
    .stSelectbox label, .stTextInput label, .stTextArea label, .stFileUploader label {
        color: #f8f9fa !important;
        font-weight: 600 !important;
        background: none !important;
        font-size: 1rem !important;
    }
    
    /* Form inputs - white background with dark text */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #ffffff !important;
        color: #212529 !important;
        border: 1px solid #ced4da !important;
    }
    
    .stTextInput input, .stTextArea textarea {
        background-color: #ffffff !important;
        color: #212529 !important;
        border: 1px solid #ced4da !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #f8f9fa !important;
        font-weight: bold !important;
    }
    
    /* Regular text */
    p, span, div {
        color: #f8f9fa !important;
    }
    
    /* Streamlit specific overrides */
    [data-testid="stForm"] label {
        color: #f8f9fa !important;
        font-weight: 600 !important;
    }
    
    .stSubheader {
        color: #f8f9fa !important;
    }
    
    .stMarkdown p, .stMarkdown span {
        color: #f8f9fa !important;
    }
    
    /* Sidebar fixes */
    [data-testid="stSidebar"] * {
        color: #fafafa !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Form Visibility Test")
st.markdown("**Testing form element visibility on dark backgrounds**")

st.subheader("📝 Single Review Analysis")

# Test form elements
review_text = st.text_area(
    "Review Input",
    placeholder="Enter the review text here...",
    height=100,
    help="Paste or type the review you want to analyze"
)

col1, col2 = st.columns(2)

with col1:
    rating = st.selectbox(
        "Rating (optional)",
        [None, 1, 2, 3, 4, 5],
        help="Select the star rating if available"
    )

with col2:
    analysis_mode = st.selectbox(
        "Analysis Mode",
        ["Quick Analysis", "Deep Analysis", "Comprehensive"],
        help="Choose the depth of analysis"
    )

business_name = st.text_input(
    "Business Name (optional)",
    placeholder="e.g., Mario's Italian Restaurant",
    help="Name of the business being reviewed"
)

uploaded_file = st.file_uploader(
    "Upload Review Screenshot (optional)",
    type=['png', 'jpg', 'jpeg'],
    help="Upload an image of the review for additional context"
)

st.divider()

# Show current values
if review_text:
    st.success(f"✅ Review text entered: {len(review_text)} characters")

if rating:
    st.info(f"ℹ️ Rating selected: {rating} stars")

if business_name:
    st.info(f"ℹ️ Business name: {business_name}")

if uploaded_file:
    st.info(f"ℹ️ File uploaded: {uploaded_file.name}")

st.markdown("---")
st.markdown("**All labels and text should be clearly visible with proper contrast!**")
