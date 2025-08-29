"""
TikTok Review Analysis Dashboard
Modern AI-Powered Fake Review Detection System

Entry Point: Run this file to start the application
Author: AI Assistant
Date: August 29, 2025
"""

import streamlit as st
import sys
import os

# Add the backend to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'frontend'))

from backend.models.qwen_classifier import QwenClassifier
from backend.models.metadata_classifier import MetadataClassifier  
from backend.utils.feature_extractor import FeatureExtractor
from backend.utils.recommendation_engine import RecommendationEngine
from frontend.components.dashboard_components import DashboardComponents

# Page configuration
st.set_page_config(
    page_title="TikTok Review Analysis", 
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main > div {
        padding-top: 1rem;
        font-family: 'Inter', sans-serif;
    }
    
    h1 {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.02em;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #343a40 0%, #495057 100%) !important;
        border: 1px solid #495057 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 0.5rem 0 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
        transition: transform 0.2s ease !important;
    }
    
    .stMetric:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4) !important;
    }
    
    .stMetric [data-testid="metric-container"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    .stMetric [data-testid="metric-container"] label {
        color: #ced4da !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stMetric [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #ffffff !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        margin: 0.5rem 0 !important;
    }
    
    .stMetric [data-testid="metric-container"] [data-testid="metric-delta"] {
        color: #adb5bd !important;
        font-size: 0.85rem !important;
    }
    
    .stTextArea textarea, .stTextInput input {
        background: #343a40 !important;
        border: 2px solid #495057 !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        padding: 1rem !important;
        transition: border-color 0.3s ease !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #0d6efd !important;
        box-shadow: 0 0 0 2px rgba(13, 110, 253, 0.25) !important;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        background: #343a40 !important;
        border: 2px solid #495057 !important;
        border-radius: 8px !important;
        color: #ffffff !important;
    }
    
    label {
        color: #f8f9fa !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #0d6efd 0%, #0b5ed7 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.75rem 2rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #0b5ed7 0%, #0a58ca 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(13, 110, 253, 0.4) !important;
    }
    
    .section-header {
        background: linear-gradient(135deg, #343a40 0%, #495057 100%);
        padding: 2rem;
        border-radius: 12px;
        margin: 2rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'classifiers_loaded' not in st.session_state:
    st.session_state.classifiers_loaded = False
    st.session_state.qwen_classifier = None
    st.session_state.metadata_classifier = None
    st.session_state.feature_extractor = None
    st.session_state.recommendation_engine = None

# Header
st.markdown("""
<div style="text-align: center; margin-bottom: 3rem;">
    <h1 style="font-size: 3rem; font-weight: 700; margin-bottom: 0.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        🎭 TikTok Review Analysis
    </h1>
    <p style="font-size: 1.2rem; color: #adb5bd; margin-bottom: 0;">
        Enhanced Multimodal Review Classification with AI-Powered Analysis
    </p>
</div>
""", unsafe_allow_html=True)

# Load classifiers
@st.cache_resource
def load_classifiers():
    """Load all AI models and classifiers"""
    with st.spinner("🤖 Loading AI models..."):
        try:
            qwen_classifier = QwenClassifier()
            metadata_classifier = MetadataClassifier()
            feature_extractor = FeatureExtractor()
            recommendation_engine = RecommendationEngine()
            
            return qwen_classifier, metadata_classifier, feature_extractor, recommendation_engine
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            return None, None, None, None

# Load models
if not st.session_state.classifiers_loaded:
    models = load_classifiers()
    if all(model is not None for model in models):
        (st.session_state.qwen_classifier, 
         st.session_state.metadata_classifier,
         st.session_state.feature_extractor,
         st.session_state.recommendation_engine) = models
        st.session_state.classifiers_loaded = True
        st.success("✅ AI models loaded successfully!")

# Main analysis section
st.markdown("""
<div class="section-header">
    <h2 style="color: #ffffff; margin: 0; font-size: 1.8rem; font-weight: 600;">
        📝 Single Review Analysis
    </h2>
    <p style="color: #adb5bd; margin: 0.5rem 0 0 0; font-size: 1rem;">
        Analyze individual reviews for authenticity and quality using AI models
    </p>
</div>
""", unsafe_allow_html=True)

# Analysis form
col1, col2 = st.columns([2, 1])

with col1:
    review_text = st.text_area(
        "Review Text",
        placeholder="Enter the review text to analyze... (e.g., 'Great food and service!')",
        height=120,
        help="Paste or type the review text you want to analyze"
    )

with col2:
    rating = st.selectbox(
        "Rating (optional)",
        [None, 1, 2, 3, 4, 5],
        format_func=lambda x: "No rating" if x is None else f"{x} ⭐"
    )
    
    business_name = st.text_input(
        "Business Name (optional)",
        placeholder="e.g., Mario's Italian Restaurant"
    )

# Analysis button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🔍 Analyze Review", type="primary")

# Analysis logic
if analyze_button and review_text and st.session_state.classifiers_loaded:
    with st.spinner("🔄 Analyzing review..."):
        try:
            # Extract features
            features = st.session_state.feature_extractor.extract_features(
                review_text, rating=rating, business_name=business_name
            )
            
            # Debug: Show extracted features
            st.write("**Debug - Features:**", features)
            
            # Get Qwen classification
            qwen_result = st.session_state.qwen_classifier.classify_review(review_text)
            
            # Debug: Show what Qwen actually returned
            st.write("**Debug - Qwen Result:**", qwen_result)
            
            # Map Qwen result to expected format
            qwen_mapped = {
                'label': qwen_result.get('category', 'legitimate').lower(),
                'confidence': qwen_result.get('confidence', 0.5),
                'reasoning': qwen_result.get('reasoning', 'No reasoning provided'),
                'method': qwen_result.get('method', 'qwen_llm')
            }
            
            # Get metadata classification
            metadata_result = st.session_state.metadata_classifier.classify_review(
                review_text, rating, business_name=business_name
            )
            
            # Debug: Show what metadata classifier returned
            st.write("**Debug - Metadata Result:**", metadata_result)
            
            # Prepare analysis result for recommendation engine
            analysis_result = {
                'qwen_analysis': qwen_mapped,
                'metadata_analysis': metadata_result,
                'features': features,
                'final_verdict': {
                    'label': qwen_mapped.get('label', 'legitimate'),
                    'confidence': qwen_mapped.get('confidence', 0.5),
                    'action': 'REMOVE' if qwen_mapped.get('label') in ['fake_review', 'spam', 'rating_manipulation'] else 'APPROVE'
                }
            }
            
            # Generate recommendations
            recommendations = st.session_state.recommendation_engine.generate_recommendations(analysis_result)
            primary_recommendation = recommendations[0] if recommendations else {
                'action': 'APPROVE',
                'reason': 'Review appears legitimate based on analysis',
                'confidence': qwen_mapped.get('confidence', 0.5)
            }
            
            # Display results
            st.markdown("""
            <div class="section-header">
                <h3 style="color: #ffffff; margin: 0; font-size: 1.5rem;">
                    📊 Analysis Results
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            # Get proper display values
            classification_label = qwen_mapped.get('label', 'legitimate').replace('_', ' ').title()
            if classification_label.lower() == 'fake review':
                classification_label = "Fake Review"
            elif classification_label.lower() == 'legitimate':
                classification_label = "Legitimate"
            elif classification_label.lower() == 'rating manipulation':
                classification_label = "Rating Manipulation"
            elif classification_label.lower() == 'spam':
                classification_label = "Spam Content"
            
            action_label = primary_recommendation.get('action', 'APPROVE').replace('_', ' ').title()
            confidence_score = qwen_mapped.get('confidence', 0.5)
            
            # Ensure confidence_score is a float
            if isinstance(confidence_score, str):
                try:
                    confidence_score = float(confidence_score)
                except:
                    confidence_score = 0.5
            
            with col1:
                st.metric(
                    "Classification", 
                    classification_label,
                    f"{confidence_score:.3f} confidence"
                )
            
            with col2:
                st.metric(
                    "Action", 
                    action_label,
                    "🔴" if action_label == "Remove" else "✅"
                )
            
            with col3:
                st.metric(
                    "Method", 
                    "Qwen + Multi",
                    "AI Enhanced"
                )
            
            with col4:
                components_used = len([x for x in [qwen_result, metadata_result, features] if x])
                st.metric(
                    "Components", 
                    f"{components_used}/3",
                    "Full Analysis"
                )
            
            # Recommendation display
            if primary_recommendation:
                action = primary_recommendation.get('action', 'REVIEW')
                reason = primary_recommendation.get('reason', 'Analysis completed')
                confidence = primary_recommendation.get('confidence', 0.0)
                
                # Ensure confidence is a float
                if isinstance(confidence, str):
                    try:
                        confidence = float(confidence)
                    except:
                        confidence = 0.0
                
                # Color coding for actions
                if action == 'REMOVE':
                    color = "#dc3545"  # Red
                    icon = "🚫"
                elif action == 'APPROVE':
                    color = "#28a745"  # Green  
                    icon = "✅"
                else:
                    color = "#ffc107"  # Yellow
                    icon = "⚠️"
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, rgba(52, 58, 64, 0.8) 0%, rgba(73, 80, 87, 0.8) 100%);
                    border-left: 4px solid {color};
                    border-radius: 8px;
                    padding: 1.5rem;
                    margin: 1rem 0;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                ">
                    <h4 style="margin: 0 0 0.5rem 0; color: #ffffff !important; font-weight: 600; font-size: 1.2rem;">
                        {icon} {action.replace('_', ' ').title()} This Review
                    </h4>
                    <p style="margin: 0.5rem 0; color: #f8f9fa !important; font-size: 1rem; line-height: 1.5;">
                        {reason}
                    </p>
                    <small style='color: #ced4da !important; font-weight: 400;'>
                        <strong>AI Reasoning:</strong> {qwen_mapped.get('reasoning', 'No reasoning provided')}<br>
                        <strong>Confidence:</strong> {confidence:.1%}
                    </small>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")

elif analyze_button and not review_text:
    st.error("⚠️ Please enter review text to analyze")

elif analyze_button and not st.session_state.classifiers_loaded:
    st.error("⚠️ AI models are not loaded. Please refresh the page.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #6c757d;">
    <p>🤖 Powered by Qwen 2.5-3B-Instruct | Enhanced Multimodal Classification</p>
    <p style="font-size: 0.9rem;">Built with Streamlit • Modern UI Design • August 2025</p>
</div>
""", unsafe_allow_html=True)
