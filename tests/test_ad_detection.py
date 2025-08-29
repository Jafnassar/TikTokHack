#!/usr/bin/env python3
"""
Test specific advertisement detection
"""

import sys
sys.path.append('.')

from backend.services.classification_service import ReviewClassificationService

print('🎯 Testing Enhanced Advertisement Detection')

service = ReviewClassificationService()

# Test the enhanced advertisement detection
ad_review = """I have a 70lb goldendoodle who is basically a walking mop and hates being groomed. I was nervous, but the team at Pampered Paws Salon was amazing! They were so patient with him and even use that calming oatmeal shampoo that left his coat unbelievably soft.

They really care about the animals here. I was thrilled that their "First-Time Groom" package included a free blueberry facial and a bandana! I heard the groomer, Sarah, is booking up weeks in advance, so I immediately pre-booked my next three appointments. Text (555) 123-PAWS to get on their list—your furry best friend will thank you"""

print('Testing advertisement review...')

result = service.classify_review(
    review_text=ad_review,
    rating=5,
    user_metadata={'review_count': 15, 'avg_rating': 4.2, 'rating_std': 0.8},
    business_name='Pampered Paws Salon'
)

verdict = result.get('final_verdict', {})
qwen_analysis = result.get('qwen_analysis', {})

print(f'Category: {verdict.get("category", "Unknown")}')
print(f'Confidence: {verdict.get("confidence", 0):.3f}')
print(f'Action: {verdict.get("action", "Unknown")}')

print('\nQwen Reasoning:')
print(qwen_analysis.get('reasoning', 'No reasoning available'))

# Test a fake review
fake_review = "This place is absolutely perfect in every way. The food is incredible and the service is flawless. Everyone should eat here immediately. 10/10 stars!"

print('\n' + '='*50)
print('Testing fake review...')

fake_result = service.classify_review(
    review_text=fake_review,
    rating=5,
    user_metadata={'review_count': 2, 'avg_rating': 5.0, 'rating_std': 0.0},
    business_name='Suspicious Eatery'
)

fake_verdict = fake_result.get('final_verdict', {})
fake_qwen = fake_result.get('qwen_analysis', {})

print(f'Category: {fake_verdict.get("category", "Unknown")}')
print(f'Confidence: {fake_verdict.get("confidence", 0):.3f}')
print(f'Action: {fake_verdict.get("action", "Unknown")}')

print('\nQwen Reasoning:')
print(fake_qwen.get('reasoning', 'No reasoning available'))
