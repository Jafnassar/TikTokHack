#!/usr/bin/env python3
"""
Feature Extractor for Review Analysis
Extracts comprehensive features from reviews for classification
"""

import re
import datetime
from typing import Dict, List, Optional, Any
import math
import pandas as pd

class FeatureExtractor:
    def __init__(self):
        """Initialize the feature extractor"""
        # Spam keywords
        self.spam_keywords = {
            'promotional': ['discount', 'coupon', 'offer', 'deal', 'sale', 'promo', 'special'],
            'generic': ['good', 'great', 'excellent', 'nice', 'amazing', 'perfect', 'awesome'],
            'suspicious': ['definitely', 'absolutely', 'totally', 'completely', 'highly recommend']
        }
        
        # Business keywords for relevance check
        self.business_keywords = {
            'restaurant': ['food', 'meal', 'taste', 'flavor', 'service', 'staff', 'menu', 'dish'],
            'location': ['location', 'parking', 'atmosphere', 'ambiance', 'seating', 'space'],
            'experience': ['experience', 'visit', 'time', 'occasion', 'recommend', 'return']
        }
        
    def extract_features(self, review_text: str, rating: Optional[int] = None, 
                        user_metadata: Optional[Dict] = None, business_name: Optional[str] = None,
                        has_images: bool = False) -> Dict[str, Any]:
        """
        Extract comprehensive features from a review
        
        Args:
            review_text (str): The review text
            rating (int, optional): Rating given (1-5)
            user_metadata (dict, optional): User behavioral metadata
            business_name (str, optional): Business name
            has_images (bool): Whether review has images
            
        Returns:
            dict: Extracted features
        """
        
        features = {}
        
        # Basic text features
        features.update(self._extract_text_features(review_text))
        
        # Rating features
        features.update(self._extract_rating_features(rating))
        
        # User behavioral features
        features.update(self._extract_user_features(user_metadata))
        
        # Content quality features
        features.update(self._extract_content_features(review_text, business_name))
        
        # Temporal features (if available)
        features.update(self._extract_temporal_features())
        
        # Image features
        features['has_images'] = has_images
        
        return features
    
    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """Extract basic text-based features"""
        if not text:
            return {
                'review_length': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'exclamation_count': 0,
                'question_count': 0,
                'uppercase_ratio': 0,
                'numeric_count': 0
            }
        
        # Basic counts
        word_count = len(text.split())
        sentence_count = len([s for s in text.split('.') if s.strip()])
        
        # Character analysis
        exclamation_count = text.count('!')
        question_count = text.count('?')
        uppercase_count = sum(1 for c in text if c.isupper())
        numeric_count = sum(1 for c in text if c.isdigit())
        
        # Ratios
        uppercase_ratio = uppercase_count / len(text) if text else 0
        avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
        
        return {
            'review_length': len(text),
            'word_count': word_count,
            'sentence_count': max(sentence_count, 1),
            'avg_word_length': avg_word_length,
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'uppercase_ratio': uppercase_ratio,
            'numeric_count': numeric_count
        }
    
    def _extract_rating_features(self, rating: Optional[int]) -> Dict[str, Any]:
        """Extract rating-based features"""
        if rating is None:
            return {
                'rating': 3,  # Default middle rating
                'is_extreme_rating': False,
                'is_positive_rating': True,
                'is_negative_rating': False
            }
        
        return {
            'rating': rating,
            'is_extreme_rating': rating in [1, 5],
            'is_positive_rating': rating >= 4,
            'is_negative_rating': rating <= 2
        }
    
    def _extract_user_features(self, user_metadata: Optional[Dict]) -> Dict[str, Any]:
        """Extract user behavioral features"""
        if not user_metadata:
            return {
                'user_review_count': 10,  # Default values
                'user_avg_rating': 3.5,
                'user_rating_std': 1.0,
                'is_new_user': False,
                'is_prolific_user': False,
                'has_consistent_ratings': True
            }
        
        review_count = user_metadata.get('review_count', 10)
        avg_rating = user_metadata.get('avg_rating', 3.5)
        rating_std = user_metadata.get('rating_std', 1.0)
        
        return {
            'user_review_count': review_count,
            'user_avg_rating': avg_rating,
            'user_rating_std': rating_std,
            'is_new_user': review_count < 5,
            'is_prolific_user': review_count > 100,
            'has_consistent_ratings': rating_std < 0.5
        }
    
    def _extract_content_features(self, text: str, business_name: Optional[str] = None) -> Dict[str, Any]:
        """Extract content quality and relevance features"""
        if not text:
            text = ""
        
        text_lower = text.lower()
        
        # Spam keyword counts
        promotional_count = sum(1 for keyword in self.spam_keywords['promotional'] if keyword in text_lower)
        generic_count = sum(1 for keyword in self.spam_keywords['generic'] if keyword in text_lower)
        suspicious_count = sum(1 for keyword in self.spam_keywords['suspicious'] if keyword in text_lower)
        
        # Business relevance
        restaurant_keywords = sum(1 for keyword in self.business_keywords['restaurant'] if keyword in text_lower)
        location_keywords = sum(1 for keyword in self.business_keywords['location'] if keyword in text_lower)
        experience_keywords = sum(1 for keyword in self.business_keywords['experience'] if keyword in text_lower)
        
        # Business name mention
        business_mentioned = False
        if business_name:
            business_mentioned = business_name.lower() in text_lower
        
        # Repetitive patterns
        words = text_lower.split()
        unique_words = len(set(words))
        word_diversity = unique_words / len(words) if words else 1.0
        
        return {
            'promotional_keywords': promotional_count,
            'generic_keywords': generic_count,
            'suspicious_keywords': suspicious_count,
            'restaurant_keywords': restaurant_keywords,
            'location_keywords': location_keywords,
            'experience_keywords': experience_keywords,
            'business_mentioned': business_mentioned,
            'word_diversity': word_diversity,
            'has_specific_details': restaurant_keywords + location_keywords > 2
        }
    
    def _extract_temporal_features(self) -> Dict[str, Any]:
        """Extract temporal features (placeholder - would need timestamp data)"""
        # In a real implementation, this would analyze:
        # - Review posting time patterns
        # - Burst detection (many reviews in short time)
        # - Seasonal patterns
        
        return {
            'posted_on_weekend': False,  # Placeholder
            'posted_late_night': False,  # Placeholder
            'part_of_burst': False      # Placeholder
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        # Generate a sample feature dict to get all keys
        sample_features = self.extract_features(
            "Sample review text", 
            rating=4, 
            user_metadata={'review_count': 10, 'avg_rating': 3.5, 'rating_std': 1.0},
            business_name="Sample Restaurant",
            has_images=False
        )
        
        return list(sample_features.keys())
    
    def features_to_vector(self, features: Dict[str, Any]) -> List[float]:
        """Convert feature dictionary to numerical vector"""
        feature_names = self.get_feature_names()
        
        vector = []
        for name in feature_names:
            value = features.get(name, 0)
            
            # Convert boolean to float
            if isinstance(value, bool):
                vector.append(float(value))
            # Convert numeric values
            elif isinstance(value, (int, float)):
                vector.append(float(value))
            else:
                # Handle unexpected types
                vector.append(0.0)
        
        return vector
    
    def batch_extract_features(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for a batch of reviews"""
        features_list = []
        
        for idx, row in reviews_df.iterrows():
            features = self.extract_features(
                review_text=row.get('text', ''),
                rating=row.get('rating', None),
                user_metadata={
                    'review_count': row.get('user_review_count', 10),
                    'avg_rating': row.get('user_avg_rating', 3.5),
                    'rating_std': row.get('user_rating_std', 1.0)
                },
                business_name=row.get('business_name', ''),
                has_images=row.get('has_photos', False)
            )
            
            features['review_id'] = idx
            features_list.append(features)
        
        return pd.DataFrame(features_list)
