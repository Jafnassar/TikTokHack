#!/usr/bin/env python3
"""
Dashboard Components for Review Analysis
Dynamic visualization components that update based on analysis results
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

class DashboardComponents:
    def __init__(self):
        """Initialize dashboard components"""
        self.colors = {
            'LEGITIMATE': '#2E8B57',      # Sea Green
            'SPAM': '#DC143C',            # Crimson
            'FAKE_REVIEW': '#FF4500',     # Orange Red
            'NO_EXPERIENCE': '#FF6347',   # Tomato
            'RATING_MANIPULATION': '#B22222',  # Fire Brick
            'REPETITIVE_SPAM': '#8B0000', # Dark Red
            'LOCATION_MISMATCH': '#FFD700',    # Gold
            'SUSPICIOUS_USER_PATTERN': '#FFA500',  # Orange
            'IMAGE_TEXT_MISMATCH': '#DAA520',      # Goldenrod
            'TEMPORAL_ANOMALY': '#F0E68C'          # Khaki
        }
        
        self.action_colors = {
            'APPROVE': '#28a745',         # Green
            'FLAG_FOR_REVIEW': '#ffc107', # Yellow
            'REMOVE': '#dc3545'           # Red
        }
    
    def render_analysis_header(self, analysis_result: Dict[str, Any]) -> None:
        """Render clean, modern analysis header"""
        verdict = analysis_result.get('final_verdict', {})
        category = verdict.get('category', 'LEGITIMATE')
        confidence = verdict.get('confidence', 0.5)
        action = verdict.get('action', 'APPROVE')
        color = verdict.get('color', '🟢')
        
        # Clean header section
        st.markdown("""
        <div style="background: linear-gradient(135deg, #343a40 0%, #495057 100%); 
                    padding: 2rem; border-radius: 12px; margin: 2rem 0; 
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);">
            <h3 style="color: #ffffff; margin: 0 0 1rem 0; font-size: 1.5rem;">
                📊 Analysis Results
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Classification",
                value=category.replace('_', ' ').title(),
                delta=f"{confidence:.3f} confidence"
            )
        
        with col2:
            st.metric(
                label="Recommended Action",
                value=action.replace('_', ' ').title(),
                delta=color
            )
        
        with col3:
            qwen_available = analysis_result.get('processing_info', {}).get('qwen_available', False)
            method = verdict.get('method', 'unknown')
            st.metric(
                label="Analysis Method",
                value="Qwen + Multi" if qwen_available else "Metadata + Image",
                delta=f"Method: {method}"
            )
        
        with col4:
            components = verdict.get('available_components', {})
            available_count = sum(components.values())
            st.metric(
                label="Components Used",
                value=f"{available_count}/3",
                delta="Qwen, Metadata, Image"
            )
    
    def render_confidence_breakdown(self, analysis_result: Dict[str, Any]) -> None:
        """Render confidence breakdown chart"""
        verdict = analysis_result.get('final_verdict', {})
        component_confidences = verdict.get('component_confidences', {})
        weights = verdict.get('combination_weights', {})
        
        if not component_confidences:
            st.warning("No confidence breakdown available")
            return
        
        # Create confidence breakdown chart
        components = []
        confidences = []
        weighted_scores = []
        
        for component, confidence in component_confidences.items():
            if component != 'image_adjustment':  # Skip image adjustment
                components.append(component.capitalize())
                confidences.append(confidence)
                weight = weights.get(component, 0)
                weighted_scores.append(confidence * weight)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Component Confidences', 'Weighted Contributions'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Raw confidences
        fig.add_trace(
            go.Bar(
                x=components,
                y=confidences,
                name="Confidence",
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(components)]
            ),
            row=1, col=1
        )
        
        # Weighted contributions
        fig.add_trace(
            go.Bar(
                x=components,
                y=weighted_scores,
                name="Weighted Score",
                marker_color=['#d62728', '#9467bd', '#8c564b'][:len(components)]
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Classification Confidence Analysis",
            height=400,
            showlegend=False
        )
        
        fig.update_yaxes(title_text="Confidence Score", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="Contribution", range=[0, 1], row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_category_distribution(self, batch_results: Optional[List[Dict[str, Any]]] = None,
                                   single_result: Optional[Dict[str, Any]] = None) -> None:
        """Render category distribution chart (dynamic based on results)"""
        
        if batch_results:
            # Batch mode - show distribution
            categories = {}
            for result in batch_results:
                category = result.get('final_verdict', {}).get('category', 'LEGITIMATE')
                categories[category] = categories.get(category, 0) + 1
            
            if not categories:
                st.warning("No categorization data available")
                return
            
            df = pd.DataFrame(list(categories.items()), columns=['Category', 'Count'])
            df['Percentage'] = (df['Count'] / df['Count'].sum()) * 100
            
            fig = px.pie(
                df, 
                values='Count', 
                names='Category',
                title=f"Review Category Distribution ({len(batch_results)} reviews)",
                color_discrete_map=self.colors
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
            
        else:
            # Single review mode - show category with confidence
            if not single_result:
                st.warning("No analysis result provided")
                return
            
            verdict = single_result.get('final_verdict', {})
            category = verdict.get('category', 'LEGITIMATE')
            confidence = verdict.get('confidence', 0.5)
            
            # Create a simple bar chart showing the single classification
            fig = go.Figure(data=[
                go.Bar(
                    x=[category.replace('_', ' ').title()],
                    y=[confidence],
                    marker_color=self.colors.get(category, '#1f77b4'),
                    text=[f'{confidence:.3f}'],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Review Classification Result",
                yaxis_title="Confidence Score",
                xaxis_title="Category",
                yaxis=dict(range=[0, 1]),
                height=400
            )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_action_recommendations(self, analysis_result: Dict[str, Any]) -> None:
        """Render clean action recommendations"""
        recommendations = analysis_result.get('recommendations', [])
        
        if not recommendations:
            st.info("No specific recommendations available")
            return
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #343a40 0%, #495057 100%); 
                    padding: 2rem; border-radius: 12px; margin: 2rem 0; 
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);">
            <h3 style="color: #ffffff; margin: 0; font-size: 1.5rem;">
                🎯 Recommendations
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Group recommendations by type
        primary_recs = [r for r in recommendations if r.get('type') == 'PRIMARY_ACTION']
        other_recs = [r for r in recommendations if r.get('type') != 'PRIMARY_ACTION']
        
        # Primary recommendations with modern styling
        for rec in primary_recs:
            action = rec.get('action', 'UNKNOWN')
            color = self.action_colors.get(action, '#6c757d')
            
            with st.container():
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, rgba(52, 58, 64, 0.8) 0%, rgba(73, 80, 87, 0.8) 100%);
                    border-left: 4px solid {color};
                    border-radius: 8px;
                    padding: 1.5rem;
                    margin: 1rem 0;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                ">
                    <h4 style="margin: 0 0 0.5rem 0; color: #ffffff !important; font-weight: 600; font-size: 1.2rem;">{rec.get('title', 'Recommendation')}</h4>
                    <p style="margin: 0.5rem 0; color: #f8f9fa !important; font-size: 1rem; line-height: 1.5;">{rec.get('description', '')}</p>
                    {f"<small style='color: #ced4da !important; font-weight: 400;'><strong>Reasoning:</strong> {rec.get('reasoning', '')}</small>" if rec.get('reasoning') else ''}
                </div>
                """, unsafe_allow_html=True)
        
        # Other recommendations in clean expandable sections
        if other_recs:
            with st.expander("📋 Additional Insights", expanded=False):
                for rec in other_recs:
                    st.markdown(f"**{rec.get('title', 'Insight')}**")
                    st.write(rec.get('description', ''))
                    
                    if rec.get('suggested_actions'):
                        st.write("Suggested actions:")
                        for action in rec['suggested_actions']:
                            st.write(f"• {action}")
                    
                    st.divider()
    
    def render_feature_importance(self, analysis_result: Dict[str, Any]) -> None:
        """Render feature importance analysis"""
        metadata_analysis = analysis_result.get('metadata_analysis', {})
        feature_importance = metadata_analysis.get('feature_importance', {})
        
        if not feature_importance:
            st.info("Feature importance data not available")
            return
        
        # Convert to DataFrame and sort
        df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance'])
        df = df.sort_values('Importance', ascending=True).tail(10)  # Top 10 features
        
        fig = px.bar(
            df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top 10 Most Important Features",
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_risk_factors(self, analysis_result: Dict[str, Any]) -> None:
        """Render risk factors analysis"""
        metadata_analysis = analysis_result.get('metadata_analysis', {})
        risk_factors = metadata_analysis.get('risk_factors', [])
        
        if not risk_factors:
            st.success("✅ No significant risk factors detected")
            return
        
        st.subheader("⚠️ Risk Factors Detected")
        
        for i, factor in enumerate(risk_factors, 1):
            st.warning(f"{i}. {factor}")
    
    def render_system_status(self, classification_service) -> None:
        """Render system status dashboard"""
        status = classification_service.get_system_status()
        
        st.subheader("🔧 System Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            qwen_status = status.get('qwen_classifier', {})
            qwen_available = qwen_status.get('available', False)
            
            st.metric(
                label="Qwen LLM",
                value="✅ Available" if qwen_available else "❌ Unavailable",
                delta=qwen_status.get('model_name', 'Unknown')
            )
        
        with col2:
            metadata_status = status.get('metadata_classifier', {})
            metadata_available = metadata_status.get('available', False)
            
            st.metric(
                label="Metadata Classifier",
                value="✅ Trained" if metadata_available else "❌ Not Trained",
                delta=f"{metadata_status.get('features_count', 0)} features"
            )
        
        with col3:
            image_status = status.get('image_analyzer', {})
            image_available = image_status.get('available', False)
            
            st.metric(
                label="Image Analyzer",
                value="✅ Available" if image_available else "❌ Unavailable",
                delta=image_status.get('model_name', 'Unknown')
            )
    
    def render_batch_summary(self, batch_results: List[Dict[str, Any]],
                           batch_recommendations: Dict[str, Any]) -> None:
        """Render batch processing summary"""
        summary = batch_recommendations.get('summary', {})
        
        st.subheader("📊 Batch Processing Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Reviews",
                value=summary.get('total_reviews', 0)
            )
        
        with col2:
            st.metric(
                label="Removal Rate",
                value=f"{summary.get('removal_rate', 0):.1f}%",
                delta="Flagged for removal"
            )
        
        with col3:
            st.metric(
                label="Approval Rate",
                value=f"{summary.get('approval_rate', 0):.1f}%",
                delta="Approved as legitimate"
            )
        
        with col4:
            st.metric(
                label="High Confidence",
                value=f"{summary.get('high_confidence_rate', 0):.1f}%",
                delta="Reliable classifications"
            )
    
    def render_comparison_chart(self, results: List[Dict[str, Any]]) -> None:
        """Render comparison chart for multiple reviews"""
        if len(results) < 2:
            st.info("Need at least 2 reviews for comparison")
            return
        
        # Extract data for comparison
        review_data = []
        for i, result in enumerate(results):
            verdict = result.get('final_verdict', {})
            review_data.append({
                'Review': f"Review {i+1}",
                'Category': verdict.get('category', 'LEGITIMATE'),
                'Confidence': verdict.get('confidence', 0.5),
                'Action': verdict.get('action', 'APPROVE')
            })
        
        df = pd.DataFrame(review_data)
        
        fig = px.scatter(
            df,
            x='Review',
            y='Confidence',
            color='Category',
            size=[0.5] * len(df),  # Fixed size
            hover_data=['Action'],
            title="Review Comparison",
            color_discrete_map=self.colors
        )
        
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
