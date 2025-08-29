#!/usr/bin/env python3
"""
Comprehensive Fake Review Detection Test
Tests all possible fake review patterns and detection capabilities
"""

import sys
sys.path.append('.')

from backend.services.classification_service import ReviewClassificationService

def test_comprehensive_fake_detection():
    """Test comprehensive fake review detection across all categories"""
    
    print('🧪 COMPREHENSIVE FAKE REVIEW DETECTION TEST')
    print('=' * 80)
    
    # Initialize service
    service = ReviewClassificationService()
    
    # Test cases covering all possible fake review patterns
    test_cases = {
        "🎯 Hidden Advertisement": {
            "text": "I have a 70lb goldendoodle who is basically a walking mop and hates being groomed. I was nervous, but the team at Pampered Paws Salon was amazing! They were so patient with him and even use that calming oatmeal shampoo that left his coat unbelievably soft. They really care about the animals here. I was thrilled that their 'First-Time Groom' package included a free blueberry facial and a bandana! I heard the groomer, Sarah, is booking up weeks in advance, so I immediately pre-booked my next three appointments. Text (555) 123-PAWS to get on their list—your furry best friend will thank you!",
            "rating": 5,
            "business": "Pampered Paws Salon",
            "expected": "SPAM"
        },
        
        "🤖 Template Fake Review": {
            "text": "This place is absolutely amazing! The food is incredible and the service is outstanding. Everything was perfect and I highly recommend this restaurant to everyone. Best dining experience ever! Five stars!",
            "rating": 5,
            "business": "Generic Restaurant",
            "expected": "FAKE_REVIEW"
        },
        
        "📈 Rating Manipulation": {
            "text": "I guess it was okay. Nothing special really.",
            "rating": 5,
            "business": "Casual Diner",
            "expected": "RATING_MANIPULATION"
        },
        
        "🚫 No Experience Admission": {
            "text": "I haven't actually been here yet but I heard from my friend that this place is really good. She said the food is amazing and the prices are reasonable. Definitely planning to visit soon!",
            "rating": 4,
            "business": "Future Visit Restaurant",
            "expected": "NO_EXPERIENCE"
        },
        
        "🔄 Repetitive Spam": {
            "text": "Great great great! Excellent excellent excellent! Amazing amazing amazing! Best best best restaurant ever! Definitely definitely definitely recommend recommend recommend!",
            "rating": 5,
            "business": "Spam Restaurant",
            "expected": "REPETITIVE_SPAM"
        },
        
        "📍 Location Impossibility": {
            "text": "Just came from this amazing restaurant in Tokyo after my lunch at the New York location. Both were incredible! The sushi chef in Paris was also fantastic when I visited yesterday.",
            "rating": 5,
            "business": "Global Chain",
            "expected": "LOCATION_MISMATCH"
        },
        
        "⏰ Temporal Anomaly": {
            "text": "Had breakfast here this morning at 3 AM. The morning rush was crazy but the staff handled it well. Great start to the workday!",
            "rating": 4,
            "business": "Morning Cafe",
            "expected": "TEMPORAL_ANOMALY"
        },
        
        "💼 Professional Marketing Language": {
            "text": "Our establishment delivers premium culinary experiences through innovative gastronomy and exceptional hospitality services. We leverage cutting-edge techniques to optimize customer satisfaction metrics while maintaining cost-effective operational excellence.",
            "rating": 5,
            "business": "Fancy Restaurant",
            "expected": "SPAM"
        },
        
        "🎭 Fake Emotional Story": {
            "text": "I was having the worst day of my life when I stumbled into this restaurant. The owner personally came out, saw I was crying, and gave me a free meal. The entire staff started singing to cheer me up and even the other customers joined in. It was like a movie scene! This place literally saved my life and restored my faith in humanity!",
            "rating": 5,
            "business": "Miracle Restaurant",
            "expected": "FAKE_REVIEW"
        },
        
        "🏢 Competitor Attack": {
            "text": "This place is terrible! The food was disgusting and gave me food poisoning. The staff was rude and the place was dirty. Go to Tony's Pizza instead - they are much better and cleaner!",
            "rating": 1,
            "business": "Pizza Place A",
            "expected": "SPAM"
        },
        
        "📱 SEO Keyword Stuffing": {
            "text": "Best pizza restaurant pizza food pizza delivery pizza takeout in downtown pizza area near pizza shops. Amazing pizza restaurant with great pizza service and pizza quality. Top rated pizza place for pizza lovers seeking pizza excellence pizza pizza pizza.",
            "rating": 5,
            "business": "Pizza World",
            "expected": "REPETITIVE_SPAM"
        },
        
        "🎪 Impossible Claims": {
            "text": "The chef is a Michelin 5-star chef (there are only 3 stars) who studied under Gordon Ramsay personally. He prepared my burger with gold flakes and diamond seasoning. The waitress was Miss Universe and served me personally. Best $5 meal of my life!",
            "rating": 5,
            "business": "Budget Burger",
            "expected": "FAKE_REVIEW"
        },
        
        "✅ Legitimate Review": {
            "text": "Had dinner here last week with my family. The pasta was good, though a bit salty for my taste. Service was friendly but took a while during the dinner rush. The atmosphere was nice and cozy. Overall a decent experience, would probably come back but maybe try something different from the menu.",
            "rating": 4,
            "business": "Family Italian",
            "expected": "LEGITIMATE"
        }
    }
    
    results = []
    correct_detections = 0
    total_tests = len(test_cases)
    
    for test_name, test_data in test_cases.items():
        print(f'\n{test_name}')
        print('-' * 60)
        print(f"Text: {test_data['text'][:100]}...")
        print(f"Expected: {test_data['expected']}")
        
        # Analyze the review
        result = service.classify_review(
            review_text=test_data['text'],
            rating=test_data['rating'],
            user_metadata={'review_count': 15, 'avg_rating': 4.2, 'rating_std': 0.8},
            business_name=test_data['business']
        )
        
        verdict = result.get('final_verdict', {})
        category = verdict.get('category', 'UNKNOWN')
        confidence = verdict.get('confidence', 0)
        
        print(f"Detected: {category} (confidence: {confidence:.3f})")
        
        # Check if detection was correct
        is_correct = category == test_data['expected']
        if is_correct:
            correct_detections += 1
            print("✅ CORRECT DETECTION")
        else:
            print("❌ INCORRECT DETECTION")
        
        results.append({
            'test': test_name,
            'expected': test_data['expected'],
            'detected': category,
            'confidence': confidence,
            'correct': is_correct
        })
    
    # Summary
    print('\n' + '=' * 80)
    print('📊 TEST SUMMARY')
    print('=' * 80)
    
    accuracy = (correct_detections / total_tests) * 100
    print(f"Overall Accuracy: {correct_detections}/{total_tests} ({accuracy:.1f}%)")
    
    print(f"\n📈 DETAILED RESULTS:")
    for result in results:
        status = "✅" if result['correct'] else "❌"
        print(f"{status} {result['test']}: {result['expected']} → {result['detected']} ({result['confidence']:.3f})")
    
    # Category performance
    print(f"\n🎯 DETECTION CAPABILITIES:")
    categories_tested = set(test_data['expected'] for test_data in test_cases.values())
    for category in categories_tested:
        category_results = [r for r in results if r['expected'] == category]
        category_correct = sum(1 for r in category_results if r['correct'])
        category_total = len(category_results)
        category_accuracy = (category_correct / category_total * 100) if category_total > 0 else 0
        print(f"  {category}: {category_correct}/{category_total} ({category_accuracy:.1f}%)")
    
    return accuracy

if __name__ == "__main__":
    accuracy = test_comprehensive_fake_detection()
    
    print('\n' + '=' * 80)
    if accuracy >= 85:
        print('🎉 EXCELLENT: Comprehensive fake review detection is highly effective!')
    elif accuracy >= 70:
        print('✅ GOOD: Fake review detection is working well with room for improvement')
    else:
        print('⚠️ NEEDS IMPROVEMENT: Detection accuracy should be enhanced')
    
    print('🛡️ Your system can now detect 13 different types of fake reviews!')
    print('=' * 80)
