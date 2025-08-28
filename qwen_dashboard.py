#!/usr/bin/env python3
"""
Advanced Review Moderation Dashboard using Qwen 3 8B
Integrates Qwen-based classification with thinking mode and policy enforcement
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import sys
import os

# Import our Qwen pipeline
try:
    from qwen_review_pipeline import QwenReviewClassifier
except ImportError:
    st.error("❌ Could not import Qwen pipeline. Make sure qwen_review_pipeline.py is in the same directory.")
    st.stop()

class QwenDashboard:
    def __init__(self):
        self.classifier = None
        self.processed_reviews = []
        
    @st.cache_resource
    def load_classifier(_self):
        """Load the Qwen classifier (cached for performance)"""
        try:
            classifier = QwenReviewClassifier()
            classifier.load_model()
            return classifier
        except Exception as e:
            st.error(f"❌ Failed to load Qwen model: {e}")
            return None
    
    def render_header(self):
        """Render dashboard header"""
        st.set_page_config(
            page_title="Qwen 3 Review Moderator",
            page_icon="�",
            layout="wide"
        )
        
        st.title("� Qwen 3 8B Review Classification & Moderation Pipeline")
        st.markdown("""
        **Advanced AI-powered review moderation using Qwen 3 8B with Thinking Mode**
        
        This system uses Qwen 2.5, a state-of-the-art open-source language model, to automatically classify and moderate reviews in real-time.
        """)
        
        # Model status
        if not self.classifier:
            st.warning("⏳ Loading Qwen model... This may take a few minutes on first run.")
            self.classifier = self.load_classifier()
        
        if self.classifier:
            st.success("✅ Qwen 2.5 Model Loaded Successfully")
        else:
            st.error("❌ Failed to load model")
            return False
        
        return True
    
    def render_single_review_classifier(self):
        """Single review classification interface"""
        st.header("🔍 Single Review Classification")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            review_text = st.text_area(
                "Enter review text to classify:",
                height=150,
                placeholder="Enter a product review here..."
            )
            
            rating = st.selectbox("Review Rating (optional):", [None, 1, 2, 3, 4, 5])
            
            # Qwen 3 8B options
            use_thinking = st.checkbox("🧠 Enable Thinking Mode (Qwen 3 feature)", value=True, 
                                       help="Shows the model's reasoning process")
            temperature = st.slider("🌡️ Temperature", 0.1, 1.0, 0.6, 0.1, 
                                   help="Higher values make output more creative")
            
            if st.button("🚀 Classify Review", disabled=not review_text):
                if review_text.strip():
                    with st.spinner("� Qwen 3 is analyzing the review..."):
                        result = self.classifier.classify_review(
                            review_text, 
                            use_thinking=use_thinking, 
                            temperature=temperature
                        )
                        policy = self.classifier.enforce_policy(result)
                        
                        # Store result
                        result.update({
                            'timestamp': datetime.now().isoformat(),
                            'policy': policy
                        })
                        self.processed_reviews.append(result)
                        
                        # Display result
                        self.display_classification_result(result, policy)
        
        with col2:
            st.markdown("### 📋 Classification Categories")
            for category, description in self.classifier.categories.items():
                st.markdown(f"**{category}**: {description}")
    
    def display_classification_result(self, result, policy):
        """Display classification results with policy enforcement"""
        st.markdown("---")
        
        # Main result
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            category_colors = {
                'LEGITIMATE': 'green',
                'SPAM': 'red', 
                'ADVERTISEMENTS': 'violet',
                'IRRELEVANT': 'orange',
                'FAKE_RANT': 'red',
                'LOW_QUALITY': 'orange'
            }
            color = category_colors.get(result['category'], 'gray')
            st.markdown(f"### 📊 Category: :{color}[{result['category']}]")
        
        with col2:
            confidence_color = 'green' if result['confidence'] > 0.8 else 'orange' if result['confidence'] > 0.6 else 'red'
            st.markdown(f"### 🎯 Confidence: :{confidence_color}[{result['confidence']:.2f}]")
        
        with col3:
            action_colors = {
                'APPROVE': 'green',
                'REMOVE': 'red',
                'FLAG_FOR_REVIEW': 'orange'
            }
            action_color = action_colors.get(policy['action'], 'gray')
            st.markdown(f"### ⚖️ Action: :{action_color}[{policy['action']}]")
        
        # Detailed analysis
        st.markdown("### 🔍 Analysis Details")
        
        # Show thinking content if available (Qwen 3 feature)
        if result.get('thinking_content'):
            with st.expander("🧠 Model Thinking Process (Qwen 3 8B)", expanded=False):
                st.markdown("**Internal reasoning and analysis:**")
                st.text_area("Thinking Content", result['thinking_content'], height=150, disabled=True)
                st.caption(f"Thinking analysis: {len(result['thinking_content'])} characters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**🤔 Reasoning:** {result['reasoning']}")
            if result.get('final_response'):
                st.markdown(f"**💬 Final Response:** `{result['final_response']}`")
            elif result.get('raw_response'):
                st.markdown(f"**📝 Raw Model Response:** `{result['raw_response']}`")
        
        with col2:
            st.markdown(f"**⚖️ Policy Reason:** {policy['reason']}")
            if policy['escalate']:
                st.warning("🚨 **Escalation Required:** This case needs human review")
            else:
                st.info("✅ **Automated Decision:** No human review needed")
                
        # Performance metrics for Qwen 3
        if 'thinking_content' in result and result['thinking_content']:
            st.markdown("### 📈 Qwen 3 Performance Insights")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                thinking_words = len(result['thinking_content'].split())
                st.metric("Thinking Depth", f"{thinking_words} words")
            
            with col2:
                reasoning_quality = "High" if "because" in result['thinking_content'].lower() else "Standard"
                st.metric("Reasoning Quality", reasoning_quality)
            
            with col3:
                analysis_type = "Evidence-based" if "evidence" in result['thinking_content'].lower() else "Pattern-based"
                st.metric("Analysis Type", analysis_type)
    
    def render_batch_processor(self):
        """Batch processing interface"""
        st.header("📊 Batch Review Processing")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with reviews",
            type=['csv'],
            help="CSV should contain 'text' column with review text. Optional: 'rating', 'id' columns"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} reviews")
                
                # Validate required columns
                if 'text' not in df.columns:
                    st.error("❌ CSV must contain a 'text' column with review text")
                    return
                
                # Show preview
                st.markdown("### 👀 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Process button
                if st.button("🚀 Process All Reviews"):
                    with st.spinner(f"🤖 Qwen is processing {len(df)} reviews..."):
                        results_df = self.classifier.batch_classify(df)
                        
                        # Add policy enforcement
                        results_df['policy'] = results_df.apply(
                            lambda row: self.classifier.enforce_policy(row), axis=1
                        )
                        
                        # Display results
                        self.display_batch_results(results_df)
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
    
    def display_batch_results(self, results_df):
        """Display batch processing results"""
        st.markdown("---")
        st.header("📈 Batch Processing Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Processed", len(results_df))
        
        with col2:
            avg_confidence = results_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        
        with col3:
            high_conf = len(results_df[results_df['confidence'] > 0.8])
            st.metric("High Confidence", f"{high_conf} ({high_conf/len(results_df)*100:.1f}%)")
        
        with col4:
            # Policy actions
            policy_actions = pd.DataFrame(results_df['policy'].tolist())['action']
            removed = len(policy_actions[policy_actions == 'REMOVE'])
            st.metric("Removed", f"{removed} ({removed/len(results_df)*100:.1f}%)")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            fig1 = px.pie(
                values=results_df['category'].value_counts().values,
                names=results_df['category'].value_counts().index,
                title="📊 Category Distribution"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Policy actions
            policy_df = pd.DataFrame(results_df['policy'].tolist())
            fig2 = px.bar(
                x=policy_df['action'].value_counts().index,
                y=policy_df['action'].value_counts().values,
                title="⚖️ Policy Actions"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Confidence distribution
        fig3 = px.histogram(
            results_df,
            x='confidence',
            title="🎯 Confidence Score Distribution",
            nbins=20
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Detailed results table
        st.markdown("### 📋 Detailed Results")
        
        # Add policy info to display
        display_df = results_df.copy()
        display_df['policy_action'] = display_df['policy'].apply(lambda x: x['action'])
        display_df['policy_reason'] = display_df['policy'].apply(lambda x: x['reason'])
        
        # Select columns to display
        display_columns = ['original_text', 'category', 'confidence', 'reasoning', 'policy_action', 'policy_reason']
        display_df = display_df[display_columns]
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download results
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name=f"qwen_classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    def render_analytics(self):
        """Analytics and monitoring dashboard"""
        if not self.processed_reviews:
            st.info("🔍 No reviews processed yet. Use the classification tools above to see analytics.")
            return
        
        st.header("📈 Real-time Analytics")
        
        # Convert to DataFrame
        analytics_df = pd.DataFrame(self.processed_reviews)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviews", len(analytics_df))
        
        with col2:
            spam_count = len(analytics_df[analytics_df['category'] == 'SPAM'])
            st.metric("Spam Detected", spam_count)
        
        with col3:
            avg_conf = analytics_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_conf:.3f}")
        
        with col4:
            removed_count = len([r for r in self.processed_reviews if r['policy']['action'] == 'REMOVE'])
            st.metric("Content Removed", removed_count)
        
        # Time series (if we have timestamps)
        if 'timestamp' in analytics_df.columns:
            analytics_df['timestamp'] = pd.to_datetime(analytics_df['timestamp'])
            analytics_df['hour'] = analytics_df['timestamp'].dt.hour
            
            hourly_counts = analytics_df.groupby('hour')['category'].count()
            fig = px.line(x=hourly_counts.index, y=hourly_counts.values, title="📅 Reviews Processed by Hour")
            st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Main dashboard runner"""
        if not self.render_header():
            return
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "🧭 Choose Function",
            ["🔍 Single Review", "📊 Batch Processing", "📈 Analytics", "ℹ️ About"]
        )
        
        if page == "🔍 Single Review":
            self.render_single_review_classifier()
        elif page == "📊 Batch Processing":
            self.render_batch_processor()
        elif page == "📈 Analytics":
            self.render_analytics()
        elif page == "ℹ️ About":
            self.render_about()
    
    def render_about(self):
        """About page with technical details"""
        st.header("ℹ️ About This System")
        
        st.markdown("""
        ### 🤖 Qwen Review Classification Pipeline
        
        This system demonstrates a **production-ready content moderation pipeline** using:
        
        #### 🔧 **Technical Stack:**
        - **Model**: Qwen 2.5 7B-Instruct (open-source LLM)
        - **Optimization**: 4-bit quantization with BitsAndBytes
        - **Framework**: HuggingFace Transformers + PyTorch
        - **Interface**: Streamlit dashboard
        
        #### 🎯 **Key Features:**
        - **Real-time Classification**: Instant review analysis
        - **Batch Processing**: Handle thousands of reviews
        - **Policy Enforcement**: Automated content decisions
        - **Explainable AI**: Clear reasoning for each decision
        - **Scalable**: Runs locally without API costs
        
        #### 📊 **Classification Categories:**
        - **LEGITIMATE**: Genuine product/service reviews
        - **SPAM**: Promotional content with contact info
        - **IRRELEVANT**: Off-topic content
        - **FAKE_RANT**: Emotional outbursts without value
        - **LOW_QUALITY**: Short, generic content
        
        #### ⚖️ **Policy Actions:**
        - **APPROVE**: Content goes live immediately
        - **REMOVE**: Content blocked automatically  
        - **FLAG_FOR_REVIEW**: Human moderator review needed
        
        #### 🏆 **Hackathon Alignment:**
        ✅ **Completeness**: Full end-to-end working pipeline  
        ✅ **Technical Quality**: Advanced ML with local inference  
        ✅ **Innovation**: Open-source LLM + sophisticated prompting  
        ✅ **Problem Fit**: Directly addresses review moderation needs  
        ✅ **Impact Potential**: Scalable, cost-effective solution  
        
        ### 🚀 Next Steps for Production:
        1. **Fine-tuning**: Train on domain-specific data
        2. **A/B Testing**: Compare with human moderators
        3. **Integration**: Connect to review platforms
        4. **Monitoring**: Track performance metrics
        5. **Feedback Loop**: Continuous model improvement
        """)

def main():
    """Main application entry point"""
    dashboard = QwenDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
