#!/usr/bin/env python3
"""
Debug Qwen model responses
"""

import sys
sys.path.append('.')

from backend.models.qwen_classifier import QwenClassifier

print('🔍 Debugging Qwen Model Response')

classifier = QwenClassifier()

if classifier.is_available():
    print('✅ Qwen model loaded successfully')
    
    # Test a simple advertisement
    ad_text = "Text (555) 123-PAWS to book your appointment now!"
    
    result = classifier.classify_review(
        review_text=ad_text,
        rating=5,
        user_metadata={'review_count': 5, 'avg_rating': 4.8, 'rating_std': 0.2},
        business_name='Pet Salon'
    )
    
    print(f'\nTest review: "{ad_text}"')
    print(f'Category: {result.get("category")}')
    print(f'Confidence: {result.get("confidence")}')
    print(f'Method: {result.get("method")}')
    print(f'Reasoning: {result.get("reasoning")}')
    
    # Show raw response for debugging
    if 'raw_response' in result:
        print(f'\nRaw Response:')
        print('=' * 50)
        print(result.get('raw_response', 'No raw response'))
        print('=' * 50)
    
else:
    print('❌ Qwen model not available')
