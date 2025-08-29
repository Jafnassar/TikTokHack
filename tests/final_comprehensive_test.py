import sys
sys.path.append('backend')

from models.qwen_classifier import QwenClassifier
import json

def run_comprehensive_test():
    print("🚀 COMPREHENSIVE FAKE REVIEW DETECTION SYSTEM TEST")
    print("=" * 60)
    
    classifier = QwenClassifier()
    
    # Test cases covering all fraud categories
    test_cases = [
        # SPAM - Advertisements
        {
            "review": "Text (555) 123-PAWS to book your appointment now! Get 20% off first visit!",
            "expected": "SPAM",
            "category": "Hidden Advertisement"
        },
        
        # FAKE_REVIEW - Template language
        {
            "review": "AMAZING! Best restaurant EVER! Everything was PERFECT! Absolutely incredible service! 5 stars!!!",
            "expected": "FAKE_REVIEW", 
            "category": "Overly Perfect Experience"
        },
        
        # RATING_MANIPULATION - Sentiment mismatch
        {
            "review": "Terrible experience. Food was cold and service was awful. Would not recommend.",
            "rating": 5,
            "expected": "FAKE_REVIEW",
            "category": "Rating-Content Mismatch"
        },
        
        # NO_EXPERIENCE - Admits no visit
        {
            "review": "Haven't actually been there yet but planning to visit soon. Looks great from the outside!",
            "expected": "NO_EXPERIENCE",
            "category": "No Actual Experience"
        },
        
        # FAKE_REVIEW - Generic content
        {
            "review": "This place is the best! Great food, great service, great atmosphere! Highly recommend to everyone!",
            "expected": "FAKE_REVIEW",
            "category": "Generic Template Language"
        },
        
        # SPAM - Marketing language
        {
            "review": "Try our new 'Premium Dining Experience' package! Book now before we're fully booked for weeks!",
            "expected": "SPAM",
            "category": "Marketing Package Promotion"
        },
        
        # LEGITIMATE - Balanced and specific
        {
            "review": "Nice restaurant with good food. Service was friendly and the atmosphere was pleasant. Had the pasta which was tasty but portion could be bigger. Would probably come back.",
            "expected": "LEGITIMATE",
            "category": "Authentic Balanced Review"
        },
        
        # FAKE_REVIEW - Competitor attack
        {
            "review": "So much better than that awful place across the street! They're terrible but this place is perfect!",
            "expected": "FAKE_REVIEW",
            "category": "Competitor Comparison Attack"
        },
        
        # SPAM - Contact information
        {
            "review": "Great service! Call us at 555-FOOD or visit our website www.bestfood.com for reservations!",
            "expected": "SPAM",
            "category": "Contact Information Spam"
        },
        
        # LEGITIMATE - Honest criticism
        {
            "review": "The food was decent but nothing special. Service was slow and the place was pretty noisy. Prices were reasonable though. Might give it another try.",
            "expected": "LEGITIMATE",
            "category": "Honest Mixed Review"
        }
    ]
    
    # Run tests
    results = {"passed": 0, "failed": 0, "total": len(test_cases)}
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test['category']}")
        print(f"Review: {test['review']}")
        if test.get('rating'):
            print(f"Rating: {test['rating']} stars")
        
        result = classifier.classify_review(
            review_text=test['review'],
            rating=test.get('rating'),
            user_metadata={"review_count": 5, "avg_rating": 3.5},
            business_name="Test Restaurant"
        )
        
        print(f"Expected: {test['expected']}")
        print(f"Got: {result['category']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Reasoning: {result['reasoning'][:150]}...")
        
        if result['category'] == test['expected']:
            print("✅ PASS")
            results['passed'] += 1
        else:
            print("❌ FAIL")
            results['failed'] += 1
        
        print("-" * 50)
    
    # Final summary
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Passed: {results['passed']}/{results['total']}")
    print(f"Failed: {results['failed']}/{results['total']}")
    print(f"Success Rate: {(results['passed']/results['total'])*100:.1f}%")
    
    if results['passed'] >= 8:  # 80% success rate
        print("🏆 EXCELLENT: Comprehensive fake review detection is working!")
    elif results['passed'] >= 6:  # 60% success rate
        print("✅ GOOD: Most fake review patterns are being detected")
    else:
        print("⚠️ NEEDS IMPROVEMENT: Detection accuracy could be better")

if __name__ == "__main__":
    run_comprehensive_test()
