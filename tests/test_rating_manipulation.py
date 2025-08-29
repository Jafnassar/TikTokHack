import sys
sys.path.append('backend')

from models.qwen_classifier import QwenClassifier

def test_rating_manipulation():
    print("🔍 Testing Rating Manipulation Detection")
    
    classifier = QwenClassifier()
    
    test_cases = [
        {
            "review": "Terrible experience. Food was cold and service was awful. Would not recommend.",
            "rating": 5,
            "description": "Negative review with 5-star rating"
        },
        {
            "review": "Amazing food! Best service ever! Absolutely perfect experience!",
            "rating": 1,
            "description": "Positive review with 1-star rating"
        },
        {
            "review": "Decent place, food was okay, nothing special but not bad either.",
            "rating": 3,
            "description": "Neutral review with appropriate rating"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test['description']} ---")
        print(f"Review: {test['review']}")
        print(f"Rating: {test['rating']} stars")
        
        result = classifier.classify_review(
            review_text=test['review'],
            rating=test['rating'],
            user_metadata="Test user metadata",
            business_name="Test Restaurant"
        )
        
        print(f"Category: {result['category']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Reasoning: {result['reasoning'][:200]}...")

if __name__ == "__main__":
    test_rating_manipulation()
