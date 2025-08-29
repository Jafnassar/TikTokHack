"""
Quick verification that the dashboard styling fixes are working
"""

import streamlit as st

st.set_page_config(page_title="Style Test", layout="wide")

# Apply the same CSS fixes
st.markdown("""
<style>
    .stMetric [data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric [data-testid="metric-container"] > div {
        color: #212529 !important;
    }
    .stMetric [data-testid="metric-container"] label {
        color: #495057 !important;
        font-weight: 600;
    }
    .stMetric [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #212529 !important;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .stMetric [data-testid="metric-container"] [data-testid="metric-delta"] {
        color: #6c757d !important;
    }
    /* Strong text contrast overrides */
    h1, h2, h3, h4, h5, h6 {
        color: #212529 !important;
        font-weight: bold !important;
    }
    p, span, div {
        color: #343a40 !important;
    }
    small {
        color: #495057 !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Dashboard Style Test")
st.write("Testing text visibility fixes")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Classification",
        value="Fake Review",
        delta="0.817 confidence"
    )

with col2:
    st.metric(
        label="Recommended Action", 
        value="Remove",
        delta="🔴"
    )

with col3:
    st.metric(
        label="Analysis Method",
        value="Qwen + Multi",
        delta="Method: qwen_plus_metadata_plus_image"
    )

with col4:
    st.metric(
        label="Components Used",
        value="2/3",
        delta="Qwen, Metadata, Image"
    )

st.success("✅ All text should now be visible with proper contrast!")
st.info("ℹ️ The CSS fixes ensure dark text on light backgrounds")
st.warning("⚠️ No more white text on white background issues")

# Test action recommendations style
st.subheader("🎯 Sample Recommendation")

st.markdown("""
<div style="
    border-left: 4px solid #dc3545;
    padding: 10px;
    margin: 10px 0;
    background-color: rgba(0,0,0,0.05);
    border-radius: 5px;
">
    <h4 style="margin: 0 0 5px 0; color: #212529 !important; font-weight: bold;">Remove This Review</h4>
    <p style="margin: 5px 0; color: #343a40 !important; font-size: 1rem;">This review appears to be fake and should be removed from the platform.</p>
    <small style='color: #495057 !important; font-weight: 500;'><strong>Reasoning:</strong> Contains template language and suspicious patterns</small>
</div>
""", unsafe_allow_html=True)
