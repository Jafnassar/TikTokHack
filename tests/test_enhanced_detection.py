import sys
sys.path.append('backend')

from models.qwen_classifier import QwenClassifier

def test_enhanced_detection():
    print("🔍 Testing Enhanced Fake Review Detection")
    
    classifier = QwenClassifier()
    
    test_cases = [
        {
            "review": "Text (555) 123-PAWS to book your appointment now!",
            "expected": "SPAM",
            "description": "Advertisement with phone number"
        },
        {
            "review": "AMAZING! Best restaurant EVER! Everything was PERFECT! 5 stars!!!",
            "expected": "FAKE_REVIEW",
            "description": "Overly enthusiastic with excessive superlatives"
        },
        {
            "review": "Terrible experience. Food was cold and service was awful. Would not recommend.",
            "rating": 5,
            "expected": "RATING_MANIPULATION",
            "description": "Negative review with 5-star rating"
        },
        {
            "review": "Haven't actually been there yet but planning to visit soon. Looks great!",
            "expected": "NO_EXPERIENCE",
            "description": "Admits no actual experience"
        },
        {
            "review": "Nice restaurant with good food. Service was friendly and the atmosphere was pleasant. Had the pasta which was tasty but portion could be bigger. Would probably come back.",
            "expected": "LEGITIMATE",
            "description": "Balanced review with specific details"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test['description']} ---")
        print(f"Review: {test['review']}")
        
        result = classifier.classify_review(
            review_text=test['review'],
            rating=test.get('rating'),
            user_metadata="Test user metadata",
            business_name="Test Restaurant"
        )
        
        print(f"Expected: {test['expected']}")
        print(f"Got: {result['category']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"✅ PASS" if result['category'] == test['expected'] else "❌ FAIL")

if __name__ == "__main__":
    test_enhanced_detection()
