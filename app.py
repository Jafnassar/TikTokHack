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
import time

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

# Import additional libraries for batch processing
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

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
    st.session_state.batch_results = None
    st.session_state.analysis_mode = 'single'

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

# Batch processing functions
def process_batch_reviews(df, text_col, rating_col, business_col, max_reviews, show_detailed):
    """Process multiple reviews from CSV with optimized batch processing"""
    with st.spinner(f"⚡ Processing {max_reviews} reviews with optimized batch processing..."):
        progress_bar = st.progress(0)
        eta_text = st.empty()
        
        # Prepare batch data
        subset_df = df.head(max_reviews)
        reviews_data = []
        
        for idx, row in subset_df.iterrows():
            review_data = {
                'index': idx,
                'review_text': str(row[text_col]),
                'rating': row[rating_col] if rating_col and rating_col in row else None,
                'business_name': str(row[business_col]) if business_col and business_col in row else None,
            }
            reviews_data.append(review_data)
        
        # Progress callback function
        def update_progress(progress, eta):
            progress_bar.progress(progress)
            if eta > 0:
                eta_text.text(f"⏱️ Estimated time remaining: {int(eta)}s")
            else:
                eta_text.text("⚡ Processing...")
        
        # Use optimized batch processing
        start_time = time.time()
        qwen_results = st.session_state.qwen_classifier.classify_batch(
            reviews_data, 
            progress_callback=update_progress
        )
        
        # Process results
        results = []
        for i, (review_data, qwen_result) in enumerate(zip(reviews_data, qwen_results)):
            try:
                # Extract features (lightweight)
                features = st.session_state.feature_extractor.extract_features(
                    review_data['review_text'], 
                    rating=review_data['rating'], 
                    business_name=review_data['business_name']
                )
                
                # Map Qwen result
                qwen_mapped = {
                    'label': qwen_result.get('category', 'legitimate').lower(),
                    'confidence': qwen_result.get('confidence', 0.5),
                    'reasoning': qwen_result.get('reasoning', 'No reasoning provided'),
                    'method': qwen_result.get('method', 'qwen_llm')
                }
                
                # Fast metadata classification (simplified for batch processing)
                metadata_result = {
                    'label': 'metadata_analysis',
                    'confidence': 0.6,
                    'method': 'metadata_fast'
                }
                
                # Create analysis result
                analysis_result = {
                    'index': review_data['index'],
                    'review_text': review_data['review_text'],
                    'rating': review_data['rating'],
                    'business_name': review_data['business_name'],
                    'qwen_analysis': qwen_mapped,
                    'metadata_analysis': metadata_result,
                    'features': features,
                    'final_verdict': {
                        'label': qwen_mapped.get('label', 'legitimate'),
                        'confidence': qwen_mapped.get('confidence', 0.5),
                        'action': 'REMOVE' if qwen_mapped.get('label') not in ['legitimate'] else 'APPROVE'
                    }
                }
                
                # Generate recommendations
                recommendations = st.session_state.recommendation_engine.generate_recommendations(analysis_result)
                analysis_result['recommendations'] = recommendations
                
                results.append(analysis_result)
                
                # Update progress
                progress_bar.progress((idx + 1) / max_reviews)
                
            except Exception as e:
                st.warning(f"⚠️ Error processing review {idx}: {str(e)}")
                continue
        
        # Store results
        st.session_state.batch_results = {
            'results': results,
            'total_processed': len(results),
            'show_detailed': show_detailed
        }
        
        progress_bar.progress(1.0)
        st.success(f"✅ Successfully processed {len(results)} reviews!")

def display_batch_results(batch_data):
    """Display batch analysis results with statistics and visualizations"""
    results = batch_data['results']
    total_processed = batch_data['total_processed']
    show_detailed = batch_data['show_detailed']
    
    if not results:
        st.warning("No results to display")
        return
    
    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <h2 style="color: #ffffff; margin: 0; font-size: 1.8rem; font-weight: 600;">
            📈 Batch Analysis Results
        </h2>
        <p style="color: #adb5bd; margin: 0.5rem 0 0 0; font-size: 1rem;">
            Comprehensive statistics and insights from your review dataset
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Overall Statistics
    st.subheader("📊 Overall Statistics")
    
    # Calculate stats
    classifications = [r['qwen_analysis']['label'] for r in results]
    actions = [r['final_verdict']['action'] for r in results]
    confidences = [r['qwen_analysis']['confidence'] for r in results]
    
    # Count classifications
    from collections import Counter
    class_counts = Counter(classifications)
    action_counts = Counter(actions)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        legitimate_count = class_counts.get('legitimate', 0)
        st.metric(
            "Legitimate Reviews",
            legitimate_count,
            f"{legitimate_count/total_processed:.1%}"
        )
    
    with col2:
        fake_categories = [
            'advertisement', 'promotional_content', 'business_owner_fake', 'competitor_attack',
            'too_generic', 'too_little_detail', 'template_language', 'unrealistic_praise',
            'copy_paste_review', 'no_actual_experience', 'hearsay_review', 'planning_to_visit',
            'location_impossible', 'rating_text_mismatch', 'extreme_rating_abuse', 
            'sentiment_contradiction', 'repetitive_spam', 'keyword_stuffing', 'contact_info_spam',
            'link_spam', 'bot_generated', 'mass_generated', 'incentivized_fake',
            'paid_review_service', 'unhelpful_content', 'irrelevant_content', 'nonsensical_text',
            'wrong_business'
        ]
        fake_count = sum(class_counts.get(cat, 0) for cat in fake_categories)
        st.metric(
            "Fake Reviews",
            fake_count,
            f"{fake_count/total_processed:.1%}"
        )
    
    with col3:
        remove_count = action_counts.get('REMOVE', 0)
        st.metric(
            "Recommended Removals",
            remove_count,
            f"{remove_count/total_processed:.1%}"
        )
    
    with col4:
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        st.metric(
            "Avg Confidence",
            f"{avg_confidence:.1%}",
            "AI Certainty"
        )
    
    # Visualizations
    st.subheader("📈 Visual Analysis")
    
    col1, col2 = st.columns(2)
    
    # Classification distribution pie chart
    with col1:
        fig_class = px.pie(
            values=list(class_counts.values()),
            names=[name.replace('_', ' ').title() for name in class_counts.keys()],
            title="Classification Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_class.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_class, width='stretch')
    
    # Action recommendations pie chart  
    with col2:
        fig_action = px.pie(
            values=list(action_counts.values()),
            names=list(action_counts.keys()),
            title="Recommended Actions",
            color_discrete_map={'APPROVE': '#28a745', 'REMOVE': '#dc3545', 'REVIEW': '#ffc107'}
        )
        fig_action.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_action, width='stretch')
    
    # Confidence distribution histogram
    st.subheader("🎯 Confidence Analysis")
    fig_conf = px.histogram(
        x=confidences,
        nbins=20,
        title="Confidence Score Distribution",
        labels={'x': 'Confidence Score', 'y': 'Number of Reviews'},
        color_discrete_sequence=['#667eea']
    )
    fig_conf.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis=dict(gridcolor='#343a40'),
        yaxis=dict(gridcolor='#343a40')
    )
    st.plotly_chart(fig_conf, use_container_width=True)
    
    # Category breakdown table
    st.subheader("📋 Category Breakdown")
    category_df = pd.DataFrame([
        {
            'Category': name.replace('_', ' ').title(),
            'Count': count,
            'Percentage': f"{count/total_processed:.1%}",
            'Avg Confidence': f"{sum(r['qwen_analysis']['confidence'] for r in results if r['qwen_analysis']['label'] == name) / count:.1%}"
        }
        for name, count in class_counts.items()
    ])
    st.dataframe(category_df, use_container_width=True)
    
    # Detailed fake review breakdown
    fake_categories = [cat for cat in class_counts.keys() if cat != 'legitimate']
    if fake_categories:
        st.subheader("🔍 Detailed Fake Review Analysis")
        
        # Group categories by type
        category_groups = {
            'Marketing/Commercial': ['advertisement', 'promotional_content', 'business_owner_fake', 'competitor_attack'],
            'Content Quality': ['too_generic', 'too_little_detail', 'template_language', 'unrealistic_praise', 'copy_paste_review'],
            'Experience Issues': ['no_actual_experience', 'hearsay_review', 'planning_to_visit', 'location_impossible'],
            'Rating Manipulation': ['rating_text_mismatch', 'extreme_rating_abuse', 'sentiment_contradiction'],
            'Spam Patterns': ['repetitive_spam', 'keyword_stuffing', 'contact_info_spam', 'link_spam'],
            'Behavioral Issues': ['bot_generated', 'mass_generated', 'incentivized_fake', 'paid_review_service'],
            'Content Problems': ['unhelpful_content', 'irrelevant_content', 'nonsensical_text', 'wrong_business'],
            'Suspicious Patterns': ['suspicious_user_pattern', 'image_text_mismatch', 'temporal_anomaly', 'unusual_language_pattern']
        }
        
        for group_name, group_categories in category_groups.items():
            group_count = sum(class_counts.get(cat, 0) for cat in group_categories)
            if group_count > 0:
                with st.expander(f"📂 {group_name} ({group_count} reviews)"):
                    group_df = pd.DataFrame([
                        {
                            'Specific Issue': cat.replace('_', ' ').title(),
                            'Count': class_counts.get(cat, 0),
                            'Percentage of Total': f"{class_counts.get(cat, 0)/total_processed:.1%}",
                            'Percentage of Group': f"{class_counts.get(cat, 0)/group_count:.1%}" if group_count > 0 else "0%"
                        }
                        for cat in group_categories if class_counts.get(cat, 0) > 0
                    ])
                    st.dataframe(group_df, use_container_width=True)
    
    # Detailed results table
    if show_detailed:
        st.subheader("🔍 Detailed Results")
        
        # Create detailed DataFrame
        detailed_data = []
        for i, result in enumerate(results):
            detailed_data.append({
                'Index': result['index'],
                'Review Text': result['review_text'][:100] + "..." if len(result['review_text']) > 100 else result['review_text'],
                'Rating': result.get('rating', 'N/A'),
                'Business': result.get('business_name', 'N/A'),
                'Classification': result['qwen_analysis']['label'].replace('_', ' ').title(),
                'Confidence': f"{result['qwen_analysis']['confidence']:.1%}",
                'Action': result['final_verdict']['action'],
                'AI Reasoning': result['qwen_analysis']['reasoning'][:50] + "..." if len(result['qwen_analysis']['reasoning']) > 50 else result['qwen_analysis']['reasoning']
            })
        
        detailed_df = pd.DataFrame(detailed_data)
        st.dataframe(detailed_df, use_container_width=True)
    
    # Export options
    st.subheader("💾 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        # Create export DataFrame
        export_data = []
        for result in results:
            export_data.append({
                'original_index': result['index'],
                'review_text': result['review_text'],
                'rating': result.get('rating', ''),
                'business_name': result.get('business_name', ''),
                'classification': result['qwen_analysis']['label'],
                'confidence': result['qwen_analysis']['confidence'],
                'action': result['final_verdict']['action'],
                'reasoning': result['qwen_analysis']['reasoning']
            })
        
        export_df = pd.DataFrame(export_data)
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="📥 Download CSV Results",
            data=csv_buffer.getvalue(),
            file_name=f"review_analysis_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Create summary report
        summary_report = f"""
# Review Analysis Summary Report

## Overview
- **Total Reviews Processed**: {total_processed}
- **Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Key Findings
- **Legitimate Reviews**: {class_counts.get('legitimate', 0)} ({class_counts.get('legitimate', 0)/total_processed:.1%})
- **Suspicious Reviews**: {fake_count} ({fake_count/total_processed:.1%})
- **Recommended Removals**: {action_counts.get('REMOVE', 0)} ({action_counts.get('REMOVE', 0)/total_processed:.1%})
- **Average Confidence**: {avg_confidence:.1%}

## Category Distribution
{chr(10).join([f"- **{name.replace('_', ' ').title()}**: {count} ({count/total_processed:.1%})" for name, count in class_counts.items()])}

## Recommendations
Based on the analysis, we recommend:
1. **Remove {action_counts.get('REMOVE', 0)} reviews** flagged as suspicious
2. **Investigate businesses** with high fake review rates
3. **Monitor patterns** in suspicious review content
4. **Consider additional verification** for low-confidence classifications
        """
        
        st.download_button(
            label="📄 Download Summary Report",
            data=summary_report,
            file_name=f"review_analysis_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )

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

# Mode selector
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("📝 Single Review Analysis", width='stretch'):
        st.session_state.analysis_mode = 'single'
        
with col2:
    if st.button("📊 Batch CSV Analysis", width='stretch'):
        st.session_state.analysis_mode = 'batch'

st.markdown("---")

# Main analysis section
if st.session_state.analysis_mode == 'single':
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
                        'action': 'REMOVE' if qwen_mapped.get('label') not in ['legitimate'] else 'APPROVE'
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
                
                # Get proper display values with enhanced categorization
                classification_label = qwen_mapped.get('label', 'legitimate').replace('_', ' ').title()
                
                # Create detailed category descriptions
                category_descriptions = {
                    'legitimate': 'Legitimate Review',
                    'advertisement': 'Advertisement (Fake)',
                    'promotional_content': 'Promotional Content (Fake)',
                    'business_owner_fake': 'Owner Pretending as Customer (Fake)',
                    'competitor_attack': 'Competitor Attack (Fake)',
                    'too_generic': 'Too Generic (Fake)',
                    'too_little_detail': 'Insufficient Detail (Fake)',
                    'template_language': 'Template Language (Fake)',
                    'unrealistic_praise': 'Unrealistic Praise (Fake)',
                    'copy_paste_review': 'Copy-Paste Review (Fake)',
                    'no_actual_experience': 'No Actual Experience (Fake)',
                    'hearsay_review': 'Hearsay Review (Fake)',
                    'planning_to_visit': 'Planning to Visit (Fake)',
                    'location_impossible': 'Location Mismatch (Fake)',
                    'rating_text_mismatch': 'Rating-Text Mismatch (Fake)',
                    'extreme_rating_abuse': 'Extreme Rating Abuse (Fake)',
                    'sentiment_contradiction': 'Sentiment Contradiction (Fake)',
                    'repetitive_spam': 'Repetitive Spam (Fake)',
                    'keyword_stuffing': 'Keyword Stuffing (Fake)',
                    'contact_info_spam': 'Contact Info Spam (Fake)',
                    'link_spam': 'Link Spam (Fake)',
                    'bot_generated': 'Bot Generated (Fake)',
                    'mass_generated': 'Mass Generated (Fake)',
                    'incentivized_fake': 'Incentivized Fake (Fake)',
                    'paid_review_service': 'Paid Review Service (Fake)',
                    'unhelpful_content': 'Unhelpful Content (Fake)',
                    'irrelevant_content': 'Irrelevant Content (Fake)',
                    'nonsensical_text': 'Nonsensical Text (Fake)',
                    'wrong_business': 'Wrong Business (Fake)',
                    'suspicious_user_pattern': 'Suspicious Pattern (Review)',
                    'image_text_mismatch': 'Image-Text Mismatch (Review)',
                    'temporal_anomaly': 'Temporal Anomaly (Review)',
                    'unusual_language_pattern': 'Unusual Language (Review)'
                }
                
                classification_display = category_descriptions.get(
                    qwen_mapped.get('label', 'legitimate').lower(), 
                    classification_label
                )
                
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
                        classification_display,
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

# Batch Analysis Mode
elif st.session_state.analysis_mode == 'batch':
    st.markdown("""
    <div class="section-header">
        <h2 style="color: #ffffff; margin: 0; font-size: 1.8rem; font-weight: 600;">
            📊 Batch CSV Analysis
        </h2>
        <p style="color: #adb5bd; margin: 0.5rem 0 0 0; font-size: 1rem;">
            Upload a CSV file to analyze multiple reviews and get comprehensive statistics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Option to use sample data or upload file
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📁 Upload Your CSV")
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="Upload a CSV file containing reviews. Required column: 'review_text' or 'text'. Optional: 'rating', 'business_name'"
        )
    
    with col2:
        st.markdown("### 🧪 Use Sample Data")
        sample_options = ["None", "simple_test.csv", "sample_reviews.csv", "test_reviews.csv"]
        selected_sample = st.selectbox(
            "Choose sample dataset",
            sample_options,
            help="Use pre-loaded sample data for testing"
        )
        
        if selected_sample != "None":
            try:
                sample_df = pd.read_csv(selected_sample)
                st.success(f"✅ Loaded {len(sample_df)} reviews from {selected_sample}")
                uploaded_file = "sample_data"  # Flag to use sample data
            except Exception as e:
                st.error(f"❌ Error loading {selected_sample}: {str(e)}")
                uploaded_file = None
    
    # Process the data
    if uploaded_file:
        try:
            # Read CSV - either uploaded file or sample data
            if uploaded_file == "sample_data":
                df = sample_df  # Use the loaded sample data
            else:
                df = pd.read_csv(uploaded_file)  # Use uploaded file
                st.success(f"✅ Loaded {len(df)} reviews from uploaded CSV")
            
            # Show first few rows
            st.subheader("📄 Data Preview")
            st.dataframe(df.head(), width='stretch')
            
            # Column mapping
            st.subheader("🔗 Column Mapping")
            cols = df.columns.tolist()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                text_col = st.selectbox(
                    "Review Text Column",
                    cols,
                    index=next((i for i, col in enumerate(cols) if 'text' in col.lower() or 'review' in col.lower()), 0)
                )
            
            with col2:
                rating_col = st.selectbox(
                    "Rating Column (optional)",
                    [None] + cols,
                    index=next((i+1 for i, col in enumerate(cols) if 'rating' in col.lower() or 'star' in col.lower()), 0)
                )
            
            with col3:
                business_col = st.selectbox(
                    "Business Name Column (optional)",
                    [None] + cols,
                    index=next((i+1 for i, col in enumerate(cols) if 'business' in col.lower() or 'name' in col.lower()), 0)
                )
            
            # Processing options
            st.subheader("⚙️ Processing Options")
            col1, col2 = st.columns(2)
            with col1:
                max_reviews = st.number_input(
                    "Max Reviews to Process",
                    min_value=1,
                    max_value=len(df),
                    value=min(100, len(df)),
                    help="Limit processing for faster results"
                )
            
            with col2:
                show_detailed = st.checkbox(
                    "Show Detailed Results",
                    value=False,
                    help="Include individual review analysis in results"
                )
            
            # Process button
            if st.button("🚀 Process Reviews", type="primary", use_container_width=True):
                if st.session_state.classifiers_loaded:
                    process_batch_reviews(df, text_col, rating_col, business_col, max_reviews, show_detailed)
                else:
                    st.error("⚠️ AI models are not loaded. Please refresh the page.")
                    
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
    
    # Show previous results if available
    if st.session_state.batch_results:
        display_batch_results(st.session_state.batch_results)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #6c757d;">
    <p>🤖 Powered by Qwen 2.5-3B-Instruct | Enhanced Multimodal Classification</p>
    <p style="font-size: 0.9rem;">Built with Streamlit • Modern UI Design • August 2025</p>
</div>
""", unsafe_allow_html=True)
