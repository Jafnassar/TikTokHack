"""
Text Contrast Comparison Test
"""

import streamlit as st

st.set_page_config(page_title="Text Contrast Test", layout="wide")

st.title("🎯 Text Contrast Comparison")

col1, col2 = st.columns(2)

with col1:
    st.subheader("❌ OLD - Gray Text (Hard to Read)")
    st.markdown("""
    <div style="
        border-left: 4px solid #dc3545;
        padding: 10px;
        margin: 10px 0;
        background-color: rgba(0,0,0,0.05);
        border-radius: 5px;
    ">
        <h4 style="margin: 0 0 5px 0; color: #666666;">Remove This Review</h4>
        <p style="margin: 5px 0; color: #888888;">This review appears to be fake and should be removed from the platform.</p>
        <small style='color: #aaaaaa;'><strong>Reasoning:</strong> Contains template language and suspicious patterns</small>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("✅ NEW - Dark Text (Easy to Read)")
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

st.divider()

st.subheader("📊 Color Values Used")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Headers (h4)**")
    st.code("#212529 - Very Dark Gray")
    st.markdown('<div style="background-color: #212529; height: 30px; border-radius: 5px;"></div>', unsafe_allow_html=True)

with col2:
    st.markdown("**Body Text (p)**")
    st.code("#343a40 - Dark Gray")
    st.markdown('<div style="background-color: #343a40; height: 30px; border-radius: 5px;"></div>', unsafe_allow_html=True)

with col3:
    st.markdown("**Small Text**")
    st.code("#495057 - Medium Gray")
    st.markdown('<div style="background-color: #495057; height: 30px; border-radius: 5px;"></div>', unsafe_allow_html=True)

st.success("✅ The new colors provide much better contrast and readability!")
st.info("ℹ️ These changes have been applied to both the style test and the main dashboard")
