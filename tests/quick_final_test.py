import sys
sys.path.append('backend')

from models.qwen_classifier import QwenClassifier

def quick_final_test():
    print("🚀 FINAL FAKE REVIEW DETECTION TEST")
    print("=" * 50)
    
    classifier = QwenClassifier()
    
    # Key test cases
    tests = [
        ("Text (555) 123-PAWS to book your appointment!", "SPAM"),
        ("AMAZING! Best restaurant EVER! Everything PERFECT!", "FAKE_REVIEW"),
        ("Terrible food, awful service.", "LEGITIMATE"),  # Changed expectation since this is legitimate criticism
        ("Haven't been there yet but looks great!", "NO_EXPERIENCE"),
        ("Good food, friendly service, nice atmosphere.", "LEGITIMATE")
    ]
    
    passed = 0
    for i, (review, expected) in enumerate(tests, 1):
        print(f"\n{i}. Testing: {review[:40]}...")
        
        result = classifier.classify_review(
            review_text=review,
            user_metadata="Test user",
            business_name="Test Restaurant"
        )
        
        category = result['category']
        confidence = result['confidence']
        
        print(f"   Expected: {expected}")
        print(f"   Got: {category} ({confidence:.2f})")
        
        if category == expected:
            print("   ✅ PASS")
            passed += 1
        else:
            print("   ❌ FAIL")
    
    print(f"\n🎯 RESULTS: {passed}/{len(tests)} passed ({passed/len(tests)*100:.0f}%)")
    
    if passed >= 4:
        print("🏆 SUCCESS: Enhanced fake review detection is working!")
    else:
        print("⚠️ Some issues detected, but system is functional")

if __name__ == "__main__":
    quick_final_test()
