#!/usr/bin/env python3
"""
Main Classification Service
Orchestrates all classification components (Qwen, Metadata, Image Analysis)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.qwen_classifier import QwenClassifier
from backend.models.metadata_classifier import MetadataClassifier
from backend.services.image_analyzer import ImageAnalyzer
from backend.utils.feature_extractor import FeatureExtractor
from backend.utils.recommendation_engine import RecommendationEngine

class ReviewClassificationService:
    def __init__(self):
        """Initialize the main classification service"""
        print("🚀 Initializing Enhanced Multimodal Review Classification Service...")
        
        # Initialize all components
        self.qwen_classifier = QwenClassifier()
        self.metadata_classifier = MetadataClassifier()
        self.image_analyzer = ImageAnalyzer()
        self.feature_extractor = FeatureExtractor()
        self.recommendation_engine = RecommendationEngine()
        
        # Classification categories
        self.categories = {
            'LEGITIMATE': {'priority': 1, 'action': 'APPROVE', 'color': '🟢'},
            'SPAM': {'priority': 2, 'action': 'REMOVE', 'color': '🔴'},
            'FAKE_REVIEW': {'priority': 3, 'action': 'REMOVE', 'color': '🔴'},
            'NO_EXPERIENCE': {'priority': 4, 'action': 'REMOVE', 'color': '🔴'},
            'RATING_MANIPULATION': {'priority': 5, 'action': 'REMOVE', 'color': '🔴'},
            'REPETITIVE_SPAM': {'priority': 6, 'action': 'REMOVE', 'color': '🔴'},
            'LOCATION_MISMATCH': {'priority': 7, 'action': 'FLAG_FOR_REVIEW', 'color': '🟡'},
            'SUSPICIOUS_USER_PATTERN': {'priority': 8, 'action': 'FLAG_FOR_REVIEW', 'color': '🟡'},
            'IMAGE_TEXT_MISMATCH': {'priority': 9, 'action': 'FLAG_FOR_REVIEW', 'color': '🟡'},
            'TEMPORAL_ANOMALY': {'priority': 10, 'action': 'FLAG_FOR_REVIEW', 'color': '🟡'}
        }
        
        print("✅ Enhanced Multimodal Review Classification Service initialized!")
    
    def classify_review(self, review_text, rating=None, user_metadata=None, 
                       business_name=None, has_images=False, image_paths=None):
        """
        Perform comprehensive multimodal review classification
        
        Args:
            review_text (str): The review text to analyze
            rating (int, optional): Star rating (1-5)
            user_metadata (dict, optional): User behavior metadata
            business_name (str, optional): Name of the business
            has_images (bool): Whether the review has images
            image_paths (list, optional): Paths to review images
            
        Returns:
            dict: Comprehensive analysis result
        """
        
        # Initialize result structure
        result = {
            'review_text': review_text,
            'rating': rating,
            'qwen_analysis': {},
            'metadata_analysis': {},
            'image_analysis': {},
            'final_verdict': {},
            'confidence_breakdown': {},
            'recommendations': [],
            'processing_info': {}
        }
        
        try:
            # 1. Extract comprehensive features
            features = self.feature_extractor.extract_features(
                review_text, rating, user_metadata, business_name, has_images
            )
            
            # 2. Qwen LLM Analysis
            print("🤖 Running Qwen LLM analysis...")
            qwen_result = self.qwen_classifier.classify_review(
                review_text, rating, user_metadata, business_name
            )
            result['qwen_analysis'] = qwen_result
            
            # 3. Metadata Analysis
            print("📊 Running metadata analysis...")
            metadata_result = self.metadata_classifier.classify_review(
                review_text, rating, user_metadata, business_name, has_images
            )
            result['metadata_analysis'] = metadata_result
            
            # 4. Image Analysis (if applicable)
            if has_images or image_paths:
                print("🖼️ Running image analysis...")
                image_features = self.image_analyzer.analyze_restaurant_images(business_name)
                
                # Image-text consistency analysis
                image_text_consistency = self.image_analyzer.analyze_image_text_consistency(
                    review_text, rating, image_features
                )
                
                # Quality indicators
                quality_indicators = self.image_analyzer.assess_image_quality_indicators(image_features)
                
                result['image_analysis'] = {
                    'image_features': image_features,
                    'image_text_consistency': image_text_consistency,
                    'image_quality_indicators': quality_indicators
                }
            
            # 5. Generate final verdict with multimodal fusion
            final_verdict = self._generate_multimodal_verdict(result)
            result['final_verdict'] = final_verdict
            
            # 6. Generate recommendations
            recommendations = self.recommendation_engine.generate_recommendations(result)
            result['recommendations'] = recommendations
            
            # 7. Add processing info
            result['processing_info'] = {
                'qwen_available': self.qwen_classifier.is_available(),
                'metadata_trained': self.metadata_classifier.is_trained,
                'image_available': self.image_analyzer.is_available(),
                'features_extracted': len(features) if features else 0
            }
            
            print(f"✅ Classification complete: {final_verdict['category']} ({final_verdict['confidence']:.3f})")
            
            return result
            
        except Exception as e:
            print(f"❌ Error in classification: {e}")
            # Return error result
            return {
                'review_text': review_text,
                'rating': rating,
                'final_verdict': {
                    'category': 'LEGITIMATE',
                    'confidence': 0.5,
                    'action': 'FLAG_FOR_REVIEW',
                    'color': '🟡',
                    'error': str(e)
                },
                'error': str(e)
            }
    
    def _generate_multimodal_verdict(self, analysis_result):
        """Generate final verdict combining all analysis components"""
        qwen_analysis = analysis_result.get('qwen_analysis', {})
        metadata_analysis = analysis_result.get('metadata_analysis', {})
        image_analysis = analysis_result.get('image_analysis', {})
        
        # Base predictions and confidences
        qwen_confidence = qwen_analysis.get('confidence', 0.5)
        qwen_category = qwen_analysis.get('category', 'LEGITIMATE')
        qwen_available = qwen_analysis.get('method') == 'qwen_llm'
        
        metadata_confidence = metadata_analysis.get('confidence', 0.5)
        metadata_prediction = metadata_analysis.get('prediction', 'legitimate')
        
        # Image consistency adjustment
        image_adjustment = 0
        if image_analysis:
            consistency = image_analysis.get('image_text_consistency', {})
            consistency_score = consistency.get('score', 0.8)
            
            if consistency_score < 0.4:
                image_adjustment = -0.15
            elif consistency_score > 0.8:
                image_adjustment = +0.05
        
        # Weighted combination based on availability
        if qwen_available:
            # Qwen is available: Qwen (60%) + Metadata (30%) + Image (10%)
            combined_confidence = (
                qwen_confidence * 0.6 + 
                metadata_confidence * 0.3 + 
                (0.8 + image_adjustment) * 0.1
            )
            
            final_category = qwen_category
            method = 'qwen_plus_metadata_plus_image'
            weights = {'qwen': 0.6, 'metadata': 0.3, 'image': 0.1}
            
        else:
            # Fallback to metadata + image: Metadata (90%) + Image (10%)
            combined_confidence = metadata_confidence * 0.9 + (0.8 + image_adjustment) * 0.1
            combined_confidence = max(0.0, min(1.0, combined_confidence))
            
            final_category = 'LEGITIMATE' if metadata_prediction == 'legitimate' else 'SUSPICIOUS_USER_PATTERN'
            method = 'metadata_plus_image'
            weights = {'qwen': 0.0, 'metadata': 0.9, 'image': 0.1}
        
        # Ensure confidence is within bounds
        combined_confidence = max(0.0, min(1.0, combined_confidence))
        
        # Get category info
        category_info = self.categories.get(final_category, self.categories['LEGITIMATE'])
        
        return {
            'category': final_category,
            'confidence': combined_confidence,
            'action': category_info['action'],
            'color': category_info['color'],
            'method': method,
            'component_confidences': {
                'qwen': qwen_confidence,
                'metadata': metadata_confidence,
                'image_adjustment': image_adjustment
            },
            'combination_weights': weights,
            'qwen_reasoning': qwen_analysis.get('reasoning', ''),
            'available_components': {
                'qwen': qwen_available,
                'metadata': metadata_analysis.get('prediction') is not None,
                'image': bool(image_analysis)
            }
        }
    
    def batch_classify_reviews(self, reviews_df, progress_callback=None):
        """Classify a batch of reviews"""
        print(f"🔍 Starting batch classification of {len(reviews_df)} reviews...")
        
        results = []
        for idx, row in reviews_df.iterrows():
            try:
                result = self.classify_review(
                    review_text=row.get('text', ''),
                    rating=row.get('rating', None),
                    user_metadata={
                        'review_count': row.get('user_review_count', 1),
                        'avg_rating': row.get('user_avg_rating', 3.0),
                        'rating_std': row.get('user_rating_std', 0.0)
                    },
                    business_name=row.get('business_name', ''),
                    has_images=row.get('has_photos', False)
                )
                
                result['review_id'] = idx
                result['original_data'] = row.to_dict()
                results.append(result)
                
                if progress_callback:
                    progress_callback(idx + 1, len(reviews_df))
                
                if (idx + 1) % 10 == 0:
                    print(f"   Processed {idx + 1}/{len(reviews_df)} reviews...")
                    
            except Exception as e:
                print(f"   Error processing review {idx}: {e}")
                continue
        
        print(f"✅ Batch classification complete: {len(results)} reviews processed")
        return results
    
    def get_system_status(self):
        """Get status of all system components"""
        return {
            'qwen_classifier': {
                'available': self.qwen_classifier.is_available(),
                'model_name': 'Qwen/Qwen2.5-3B-Instruct' if self.qwen_classifier.is_available() else 'Not loaded'
            },
            'metadata_classifier': {
                'available': self.metadata_classifier.is_trained,
                'features_count': len(self.metadata_classifier.feature_names) if self.metadata_classifier.feature_names else 0
            },
            'image_analyzer': {
                'available': self.image_analyzer.is_available(),
                'model_name': 'CLIP ViT-B/32' if self.image_analyzer.is_available() else 'Not loaded'
            }
        }
