#!/usr/bin/env python3
"""
Script to add metadata columns to sample CSV for testing
"""
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def calculate_review_metadata(review_text, rating):
    """Calculate metadata features for a review"""
    
    # Basic text features
    review_length = len(review_text)
    exclamation_count = review_text.count('!')
    caps_ratio = sum(1 for c in review_text if c.isupper()) / len(review_text) if review_text else 0
    
    # Simple sentiment score based on rating and words
    positive_words = ['amazing', 'excellent', 'great', 'fantastic', 'wonderful', 'perfect', 'best', 'love']
    negative_words = ['terrible', 'awful', 'horrible', 'worst', 'bad', 'hate', 'disgusting']
    
    text_lower = review_text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    # Sentiment score based on rating and word sentiment
    base_sentiment = (rating - 1) / 4  # Normalize rating to 0-1
    word_sentiment = (pos_count - neg_count) * 0.1
    sentiment_score = max(0, min(1, base_sentiment + word_sentiment))
    
    return {
        'review_length': review_length,
        'exclamation_count': exclamation_count,
        'caps_ratio': round(caps_ratio, 3),
        'sentiment_score': round(sentiment_score, 3)
    }

def generate_user_metadata():
    """Generate realistic user metadata"""
    # User review patterns
    user_types = [
        {'review_count': random.randint(1, 5), 'avg_rating': random.uniform(2.5, 5.0), 'days_since': random.randint(30, 365)},  # New user
        {'review_count': random.randint(6, 25), 'avg_rating': random.uniform(3.0, 4.5), 'days_since': random.randint(200, 800)},  # Regular user
        {'review_count': random.randint(50, 200), 'avg_rating': random.uniform(3.5, 4.2), 'days_since': random.randint(500, 2000)},  # Power user
    ]
    
    user_type = random.choice(user_types)
    
    return {
        'user_review_count': user_type['review_count'],
        'user_avg_rating': round(user_type['avg_rating'], 1),
        'days_since_account_creation': user_type['days_since']
    }

def generate_business_metadata():
    """Generate realistic business metadata"""
    # Business profiles
    business_types = [
        {'review_count': random.randint(50, 150), 'avg_rating': random.uniform(3.8, 4.2), 'deviation': random.uniform(0.1, 0.3)},  # Good restaurant
        {'review_count': random.randint(200, 800), 'avg_rating': random.uniform(4.0, 4.5), 'deviation': random.uniform(0.2, 0.4)},  # Popular restaurant
        {'review_count': random.randint(20, 80), 'avg_rating': random.uniform(2.5, 3.5), 'deviation': random.uniform(0.4, 0.8)},  # Struggling restaurant
    ]
    
    business_type = random.choice(business_types)
    
    return {
        'business_review_count': business_type['review_count'],
        'business_avg_rating': round(business_type['avg_rating'], 1),
        'business_rating_deviation': round(business_type['deviation'], 1)
    }

def generate_temporal_metadata():
    """Generate realistic temporal metadata"""
    # Random hour (business hours weighted)
    business_hours = list(range(11, 22))  # 11 AM to 9 PM
    other_hours = list(range(0, 11)) + list(range(22, 24))
    
    review_hour = random.choice(business_hours + business_hours + other_hours)  # Weight business hours
    
    # Weekend flag
    is_weekend = random.choice([0, 0, 0, 0, 0, 1, 1])  # 2/7 chance of weekend
    
    # Business hours flag
    is_business_hours = 1 if 11 <= review_hour <= 21 else 0
    
    return {
        'review_hour': review_hour,
        'is_weekend': is_weekend,
        'is_business_hours': is_business_hours
    }

def main():
    # Read the current CSV
    df = pd.read_csv('sample_reviews.csv')
    
    # Check if we need to add metadata
    if 'review_length' in df.columns:
        print("✅ Metadata columns already exist")
        return
    
    print("🔄 Adding metadata columns to sample CSV...")
    
    # Generate metadata for each row
    for idx, row in df.iterrows():
        review_text = row['review_text']
        rating = row['rating']
        
        # Calculate all metadata
        text_meta = calculate_review_metadata(review_text, rating)
        user_meta = generate_user_metadata()
        business_meta = generate_business_metadata()
        temporal_meta = generate_temporal_metadata()
        
        # Add to dataframe
        for key, value in {**text_meta, **user_meta, **business_meta, **temporal_meta}.items():
            df.at[idx, key] = value
    
    # Save updated CSV
    df.to_csv('sample_reviews.csv', index=False)
    print("✅ Sample CSV updated with metadata columns")
    print(f"📊 Columns: {list(df.columns)}")

if __name__ == "__main__":
    main()
