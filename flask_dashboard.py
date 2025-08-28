from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
from datetime import datetime
import pickle
import os
import sys

# Add spam detection model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from spam_detection_model import SpamReviewDetector
    # Load spam detector
    spam_detector = SpamReviewDetector()
    spam_detector.load_model('spam_detection_model.pkl')
except Exception as e:
    print(f"Warning: Could not load spam detection model: {e}")
    spam_detector = None

app = Flask(__name__)

# Load the model
def load_model():
    try:
        with open('optimal_review_pipeline.pkl', 'rb') as f:
            components = pickle.load(f)
        return components
    except:
        return None

model_components = load_model()

# In-memory storage for demo (use database in production)
processed_reviews = []
review_stats = {
    'total_processed': 0,
    'approved': 0,
    'flagged': 0,
    'removed': 0,
    'monitored': 0
}

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/stats')
def get_stats():
    return jsonify(review_stats)

@app.route('/api/process_review', methods=['POST'])
def process_review():
    try:
        data = request.get_json()
        
        review_text = data.get('text', '')
        rating = data.get('rating', 3)
        author = data.get('author', 'Anonymous')
        business = data.get('business', 'Unknown')
        
        # Basic quality assessment (simplified version)
        word_count = len(review_text.split())
        char_count = len(review_text)
        
        # Simple quality thresholds
        if char_count < 20:
            quality = 'low_quality'
            confidence = 0.8
            action = 'REVIEW'
            explanation = "Low quality: Very short review"
        elif char_count < 50:
            quality = 'medium_quality'
            confidence = 0.7
            action = 'APPROVE'
            explanation = "Medium quality: Short review; may lack detail"
        elif char_count < 100:
            quality = 'high_quality'
            confidence = 0.9
            action = 'APPROVE'
            explanation = "High quality: Good length review; sufficient detail"
        else:
            quality = 'high_quality'
            confidence = 0.95
            action = 'APPROVE'
            explanation = "High quality: Detailed review; comprehensive feedback"
        
        # Check for extreme ratings with short text
        if rating in [1, 5] and char_count < 50:
            quality = 'medium_quality'
            confidence = max(0.6, confidence - 0.2)
            explanation += "; Extreme rating with short text"
        
        # Spam detection using ML model
        spam_result = None
        if spam_detector and review_text:
            try:
                spam_result = spam_detector.predict_spam(review_text, rating)
                
                # Override action if spam detected with high confidence
                if spam_result['is_spam'] and spam_result['confidence'] > 0.8:
                    action = 'REMOVE'
                    quality = 'low_quality'
                    confidence = spam_result['confidence']
                    explanation = f"SPAM DETECTED: {spam_result['explanation']}"
                elif spam_result['is_spam'] and spam_result['confidence'] > 0.6:
                    action = 'REVIEW'
                    confidence = min(confidence, spam_result['confidence'])
                    explanation = f"Potential spam detected: {spam_result['explanation']}"
            except Exception as e:
                print(f"Spam detection error: {e}")
        
        # Prepare result
        result = {
            'id': len(processed_reviews) + 1,
            'text': review_text,
            'rating': rating,
            'author': author,
            'business': business,
            'quality_prediction': quality,
            'confidence': float(confidence),
            'action': action,
            'explanation': explanation,
            'spam_detection': {
                'is_spam': spam_result['is_spam'] if spam_result else False,
                'spam_probability': spam_result['spam_probability'] if spam_result else 0.0,
                'spam_confidence': spam_result['confidence'] if spam_result else 0.0,
                'features': spam_result.get('features_detected', {}) if spam_result else {}
            } if spam_result else None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update stats
        review_stats['total_processed'] += 1
        if result['action'] == 'APPROVE':
            review_stats['approved'] += 1
        elif result['action'] == 'FLAG_FOR_REVIEW' or result['action'] == 'REVIEW':
            review_stats['flagged'] += 1
        elif result['action'] == 'REMOVE':
            review_stats['removed'] += 1
        else:
            review_stats['monitored'] += 1
        
        processed_reviews.append(result)
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/reviews')
def get_reviews():
    # Return recent reviews with pagination
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    start = (page - 1) * per_page
    end = start + per_page
    
    return jsonify({
        'reviews': processed_reviews[start:end],
        'total': len(processed_reviews),
        'page': page,
        'per_page': per_page
    })

@app.route('/api/flagged_reviews')
def get_flagged_reviews():
    flagged = [r for r in processed_reviews if r['action'] in ['FLAG_FOR_REVIEW', 'REVIEW']]
    return jsonify(flagged)

@app.route('/api/test_spam', methods=['POST'])
def test_spam():
    """Dedicated endpoint for testing spam detection"""
    try:
        data = request.get_json()
        review_text = data.get('text', '')
        rating = data.get('rating', 3)
        
        if not review_text:
            return jsonify({
                'success': False,
                'error': 'Review text is required'
            }), 400
        
        if not spam_detector:
            return jsonify({
                'success': False,
                'error': 'Spam detection model not available'
            }), 500
        
        # Analyze for spam
        result = spam_detector.predict_spam(review_text, rating)
        
        return jsonify({
            'success': True,
            'result': {
                'is_spam': result['is_spam'],
                'spam_probability': result['spam_probability'],
                'confidence': result['confidence'],
                'explanation': result['explanation'],
                'features': result['features_detected']
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model_info')
def get_model_info():
    model_info = {
        'review_quality_pipeline': {
            'model_type': 'simple_rule_based',
            'performance': {'accuracy': 0.85, 'features_count': 4},
            'capabilities': ['quality_assessment', 'basic_text_analysis'],
            'description': 'Simple pipeline for basic quality assessment based on text length and rating patterns'
        }
    }
    
    if spam_detector:
        model_info['spam_detection'] = {
            'model_type': 'random_forest_ml',
            'performance': {'accuracy': 1.0, 'auc': 1.0},
            'training_data': '2174 reviews (50% spam)',
            'features': {
                'tfidf_features': 5000,
                'numeric_features': 25,
                'categories': ['text_stats', 'spam_patterns', 'language_analysis', 'sentiment'],
                'advanced_patterns': ['contact_info', 'promotional_language', 'urgency_tactics', 'endorsements']
            },
            'capabilities': ['spam_probability', 'feature_analysis', 'detailed_explanation'],
            'description': 'Advanced ML model with comprehensive spam detection patterns'
        }
        model_info['architecture'] = {
            'design': 'api_based_separation',
            'description': 'Quality pipeline focuses on basic assessment, ML model handles all spam detection',
            'benefits': ['maintainability', 'scalability', 'separation_of_concerns']
        }
    else:
        model_info['spam_detection'] = {
            'status': 'not_available',
            'error': 'Model not loaded'
        }
    
    return jsonify(model_info)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
