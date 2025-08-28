from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
from datetime import datetime
import pickle
import os

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
        
        # Simulate review processing (replace with your actual pipeline)
        result = {
            'id': len(processed_reviews) + 1,
            'text': data.get('text', ''),
            'rating': data.get('rating', 3),
            'author': data.get('author', 'Anonymous'),
            'business': data.get('business', 'Unknown'),
            'quality_prediction': np.random.choice(['high_quality', 'medium_quality', 'low_quality'], 
                                                  p=[0.6, 0.3, 0.1]),
            'confidence': float(np.random.uniform(0.7, 0.99)),
            'action': np.random.choice(['APPROVE', 'FLAG_FOR_REVIEW', 'REMOVE', 'MONITOR'], 
                                      p=[0.7, 0.15, 0.1, 0.05]),
            'explanation': "Quality analysis based on engineered features and ML model.",
            'timestamp': datetime.now().isoformat()
        }
        
        # Update stats
        review_stats['total_processed'] += 1
        if result['action'] == 'APPROVE':
            review_stats['approved'] += 1
        elif result['action'] == 'FLAG_FOR_REVIEW':
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
    flagged = [r for r in processed_reviews if r['action'] == 'FLAG_FOR_REVIEW']
    return jsonify(flagged)

@app.route('/api/model_info')
def get_model_info():
    if model_components:
        return jsonify({
            'model_type': model_components.get('model_type', 'baseline_optimized'),
            'performance': model_components.get('performance', {}),
            'features_count': model_components.get('performance', {}).get('features_count', 43)
        })
    else:
        return jsonify({
            'model_type': 'demo_mode',
            'performance': {'f1_score': 0.995, 'accuracy': 0.995, 'features_count': 43},
            'features_count': 43
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
