#!/usr/bin/env python3
"""
Image Analysis Service
Handles CLIP-based image analysis for review-image consistency
"""

import numpy as np
import torch
from PIL import Image
import cv2
import os
import glob
import warnings
warnings.filterwarnings('ignore')

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

class ImageAnalyzer:
    def __init__(self):
        """Initialize CLIP model for image analysis"""
        self.model = None
        self.preprocess = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Image categories and their expected characteristics
        self.image_categories = {
            'taste': {
                'keywords': ['delicious food', 'tasty meal', 'appetizing dish', 'fresh ingredients'],
                'expected_sentiment': 'positive'
            },
            'menu': {
                'keywords': ['menu board', 'food menu', 'restaurant menu', 'price list'],
                'expected_sentiment': 'neutral'
            },
            'indoor_atmosphere': {
                'keywords': ['restaurant interior', 'dining room', 'cozy atmosphere', 'nice ambiance'],
                'expected_sentiment': 'positive'
            },
            'outdoor_atmosphere': {
                'keywords': ['outdoor seating', 'patio dining', 'terrace restaurant', 'outdoor tables'],
                'expected_sentiment': 'positive'
            }
        }
        
        self.initialize_clip()
    
    def initialize_clip(self):
        """Initialize CLIP model"""
        if CLIP_AVAILABLE:
            try:
                self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
                print(f"✅ CLIP model loaded on {self.device}")
            except Exception as e:
                print(f"❌ Error loading CLIP: {e}")
                self.model = None
        else:
            print("⚠️ CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
            self.model = None
    
    def analyze_restaurant_images(self, restaurant_name=None):
        """Analyze restaurant images for context and legitimacy indicators"""
        image_features = {}
        
        if not self.model:
            return image_features
        
        # Analyze categorized images
        for category in self.image_categories.keys():
            category_path = f"data/Google Map Reviews/dataset/{category}"
            
            if os.path.exists(category_path):
                image_files = glob.glob(f"{category_path}/*.png")[:5]  # Limit for efficiency
                
                category_scores = []
                for img_path in image_files:
                    try:
                        score = self._analyze_single_image(img_path, self.image_categories[category])
                        category_scores.append(score)
                    except Exception:
                        continue
                
                if category_scores:
                    image_features[f'{category}_avg_score'] = np.mean(category_scores)
                    image_features[f'{category}_image_count'] = len(category_scores)
                else:
                    image_features[f'{category}_avg_score'] = 0.5
                    image_features[f'{category}_image_count'] = 0
        
        # Analyze specific restaurant images if provided
        if restaurant_name:
            restaurant_path = f"data/Google Map Reviews/sepetcioglu_restaurant"
            if os.path.exists(restaurant_path):
                restaurant_images = glob.glob(f"{restaurant_path}/*.png")[:10]
                
                restaurant_scores = []
                for img_path in restaurant_images:
                    try:
                        score = self._analyze_restaurant_quality(img_path)
                        restaurant_scores.append(score)
                    except Exception:
                        continue
                
                if restaurant_scores:
                    image_features['restaurant_quality_score'] = np.mean(restaurant_scores)
                    image_features['restaurant_image_count'] = len(restaurant_scores)
        
        return image_features
    
    def analyze_image_text_consistency(self, review_text, rating, image_features):
        """Analyze consistency between images and text/rating"""
        consistency_score = 0.8  # Default neutral
        
        if not image_features:
            return {'score': consistency_score, 'details': 'No images to analyze'}
        
        # Check if positive review matches image quality
        positive_words = ['great', 'excellent', 'amazing', 'love', 'best', 'wonderful']
        negative_words = ['terrible', 'awful', 'worst', 'hate', 'bad']
        
        positive_count = sum(1 for word in positive_words if word in review_text.lower())
        negative_count = sum(1 for word in negative_words if word in review_text.lower())
        
        avg_image_quality = np.mean([
            image_features.get('taste_avg_score', 0.5),
            image_features.get('indoor_atmosphere_avg_score', 0.5),
            image_features.get('outdoor_atmosphere_avg_score', 0.5),
            image_features.get('restaurant_quality_score', 0.5)
        ])
        
        # High rating + positive words should match high image quality
        if rating and rating >= 4 and positive_count > negative_count:
            if avg_image_quality < 0.4:
                consistency_score = 0.3  # Inconsistent
            elif avg_image_quality > 0.7:
                consistency_score = 0.9  # Very consistent
        
        # Low rating + negative words should match lower image quality
        elif rating and rating <= 2 and negative_count > positive_count:
            if avg_image_quality > 0.7:
                consistency_score = 0.3  # Inconsistent
            elif avg_image_quality < 0.4:
                consistency_score = 0.9  # Consistent
        
        return {
            'score': consistency_score,
            'avg_image_quality': avg_image_quality,
            'text_sentiment': 'positive' if positive_count > negative_count else 'negative',
            'details': f"Image quality: {avg_image_quality:.2f}, Consistency: {consistency_score:.2f}"
        }
    
    def assess_image_quality_indicators(self, image_features):
        """Assess various quality indicators from images"""
        indicators = {}
        
        if image_features:
            # Food presentation quality
            indicators['food_presentation'] = image_features.get('taste_avg_score', 0.5)
            
            # Atmosphere quality
            indoor_score = image_features.get('indoor_atmosphere_avg_score', 0.5)
            outdoor_score = image_features.get('outdoor_atmosphere_avg_score', 0.5)
            indicators['atmosphere_quality'] = (indoor_score + outdoor_score) / 2
            
            # Menu professionalism
            indicators['menu_quality'] = image_features.get('menu_avg_score', 0.5)
            
            # Overall establishment quality
            indicators['establishment_quality'] = image_features.get('restaurant_quality_score', 0.5)
            
            # Image completeness (how many categories have images)
            categories_with_images = sum(1 for key, value in image_features.items() 
                                       if key.endswith('_image_count') and value > 0)
            indicators['image_completeness'] = categories_with_images / 4  # 4 main categories
        
        return indicators
    
    def _analyze_single_image(self, image_path, category_info):
        """Analyze a single image using CLIP"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0)
            
            # Tokenize text descriptions
            text_inputs = clip.tokenize(category_info['keywords'])
            
            # Calculate similarities
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_inputs)
                
                # Calculate similarity scores
                similarities = torch.cosine_similarity(image_features, text_features)
                max_similarity = torch.max(similarities).item()
                
                # Normalize to 0-1 range
                score = (max_similarity + 1) / 2
                
            return score
            
        except Exception:
            return 0.5  # Default neutral score
    
    def _analyze_restaurant_quality(self, image_path):
        """Analyze restaurant image for quality indicators"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0)
            
            # Quality indicators
            quality_descriptions = [
                "high quality restaurant",
                "clean restaurant interior", 
                "professional food presentation",
                "well maintained establishment",
                "attractive restaurant design"
            ]
            
            text_inputs = clip.tokenize(quality_descriptions)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_inputs)
                
                similarities = torch.cosine_similarity(image_features, text_features)
                avg_quality_score = torch.mean(similarities).item()
                
                # Normalize to 0-1 range
                score = (avg_quality_score + 1) / 2
                
            return score
            
        except Exception:
            return 0.5
    
    def is_available(self):
        """Check if CLIP model is available"""
        return self.model is not None
