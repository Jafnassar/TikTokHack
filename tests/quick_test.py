#!/usr/bin/env python3
"""
Quick system verification test
"""

import sys
sys.path.append('.')

try:
    from backend.services.classification_service import ReviewClassificationService
    
    print('🧪 Quick System Test...')
    
    service = ReviewClassificationService()
    
    # Test a simple review
    result = service.classify_review(
        review_text='Great restaurant with amazing food!',
        rating=5,
        user_metadata={'review_count': 10, 'avg_rating': 4.0, 'rating_std': 1.0},
        business_name='Test Restaurant'
    )
    
    verdict = result.get('final_verdict', {})
    print(f'✅ System working: {verdict.get("category", "Unknown")} ({verdict.get("confidence", 0):.3f})')
    
    # Test advertisement detection
    ad_result = service.classify_review(
        review_text='Text (555) 123-PAWS to book your appointment now!',
        rating=5,
        user_metadata={'review_count': 5, 'avg_rating': 4.8, 'rating_std': 0.2},
        business_name='Pet Salon'
    )
    
    ad_verdict = ad_result.get('final_verdict', {})
    print(f'✅ Ad detection: {ad_verdict.get("category", "Unknown")} ({ad_verdict.get("confidence", 0):.3f})')
    
    print('🎉 System verification complete!')

except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
