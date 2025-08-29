#!/usr/bin/env python3
"""
Recommendation Engine for Review Analysis
Generates actionable recommendations based on classification results
"""

from typing import Dict, List, Any, Optional
import datetime

class RecommendationEngine:
    def __init__(self):
        """Initialize the recommendation engine"""
        self.action_priorities = {
            'REMOVE': 1,
            'FLAG_FOR_REVIEW': 2,
            'APPROVE': 3
        }
        
        self.confidence_thresholds = {
            'high': 0.85,
            'medium': 0.65,
            'low': 0.45
        }
    
    def generate_recommendations(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate comprehensive recommendations based on analysis results
        
        Args:
            analysis_result (dict): Complete analysis result from classification service
            
        Returns:
            list: List of recommendation dictionaries
        """
        recommendations = []
        
        final_verdict = analysis_result.get('final_verdict', {})
        qwen_analysis = analysis_result.get('qwen_analysis', {})
        metadata_analysis = analysis_result.get('metadata_analysis', {})
        image_analysis = analysis_result.get('image_analysis', {})
        
        # Primary action recommendation
        primary_rec = self._generate_primary_recommendation(final_verdict)
        recommendations.append(primary_rec)
        
        # Confidence-based recommendations
        confidence_recs = self._generate_confidence_recommendations(final_verdict)
        recommendations.extend(confidence_recs)
        
        # Component-specific recommendations
        if qwen_analysis:
            qwen_recs = self._generate_qwen_recommendations(qwen_analysis)
            recommendations.extend(qwen_recs)
        
        if metadata_analysis:
            metadata_recs = self._generate_metadata_recommendations(metadata_analysis)
            recommendations.extend(metadata_recs)
        
        if image_analysis:
            image_recs = self._generate_image_recommendations(image_analysis)
            recommendations.extend(image_recs)
        
        # Business intelligence recommendations
        business_recs = self._generate_business_recommendations(analysis_result)
        recommendations.extend(business_recs)
        
        # Sort by priority
        recommendations.sort(key=lambda x: x.get('priority', 5))
        
        return recommendations
    
    def _generate_primary_recommendation(self, final_verdict: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the primary action recommendation"""
        category = final_verdict.get('category', 'LEGITIMATE')
        confidence = final_verdict.get('confidence', 0.5)
        action = final_verdict.get('action', 'APPROVE')
        
        if action == 'REMOVE':
            if confidence > self.confidence_thresholds['high']:
                return {
                    'type': 'PRIMARY_ACTION',
                    'action': 'REMOVE',
                    'priority': 1,
                    'confidence': 'HIGH',
                    'title': '🚫 Remove Review',
                    'description': f'High confidence ({confidence:.3f}) detection of {category}. Immediate removal recommended.',
                    'reasoning': f'Multiple signals indicate this is a {category.lower().replace("_", " ")} review.',
                    'automated': True
                }
            else:
                return {
                    'type': 'PRIMARY_ACTION',
                    'action': 'FLAG_FOR_REVIEW',
                    'priority': 2,
                    'confidence': 'MEDIUM',
                    'title': '🟡 Flag for Human Review',
                    'description': f'Medium confidence ({confidence:.3f}) detection of {category}. Human review recommended.',
                    'reasoning': 'Confidence level suggests manual verification before action.',
                    'automated': False
                }
        
        elif action == 'FLAG_FOR_REVIEW':
            return {
                'type': 'PRIMARY_ACTION',
                'action': 'FLAG_FOR_REVIEW',
                'priority': 2,
                'confidence': 'MEDIUM',
                'title': '🔍 Manual Review Required',
                'description': f'Potential {category} detected. Human verification needed.',
                'reasoning': 'Ambiguous signals require human judgment.',
                'automated': False
            }
        
        else:  # APPROVE
            return {
                'type': 'PRIMARY_ACTION',
                'action': 'APPROVE',
                'priority': 3,
                'confidence': 'HIGH' if confidence > self.confidence_thresholds['high'] else 'MEDIUM',
                'title': '✅ Approve Review',
                'description': f'Review appears legitimate (confidence: {confidence:.3f}).',
                'reasoning': 'No significant indicators of fraudulent content.',
                'automated': True
            }
    
    def _generate_confidence_recommendations(self, final_verdict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on confidence levels"""
        confidence = final_verdict.get('confidence', 0.5)
        recommendations = []
        
        if confidence < self.confidence_thresholds['low']:
            recommendations.append({
                'type': 'CONFIDENCE_WARNING',
                'priority': 2,
                'title': '⚠️ Low Confidence Detection',
                'description': f'Classification confidence is low ({confidence:.3f}). Consider additional verification.',
                'reasoning': 'Low confidence may indicate edge case or insufficient training data.',
                'suggested_actions': [
                    'Request additional review signals',
                    'Cross-reference with user history',
                    'Manual expert review'
                ]
            })
        
        elif confidence > self.confidence_thresholds['high']:
            recommendations.append({
                'type': 'CONFIDENCE_HIGH',
                'priority': 4,
                'title': '🎯 High Confidence Classification',
                'description': f'Very confident classification ({confidence:.3f}). Safe for automated action.',
                'reasoning': 'Multiple strong signals align for reliable classification.',
                'suggested_actions': ['Proceed with automated action']
            })
        
        return recommendations
    
    def _generate_qwen_recommendations(self, qwen_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on Qwen LLM analysis"""
        recommendations = []
        
        if qwen_analysis.get('method') == 'qwen_llm':
            reasoning = qwen_analysis.get('reasoning', '')
            confidence = qwen_analysis.get('confidence', 0.5)
            
            if 'training data' in reasoning.lower():
                recommendations.append({
                    'type': 'QWEN_INSIGHT',
                    'priority': 3,
                    'title': '🤖 AI-Enhanced Detection',
                    'description': 'Qwen LLM analysis incorporates patterns from 8.7M+ review training examples.',
                    'reasoning': reasoning,
                    'confidence_score': confidence
                })
        
        return recommendations
    
    def _generate_metadata_recommendations(self, metadata_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on metadata analysis"""
        recommendations = []
        
        risk_factors = metadata_analysis.get('risk_factors', [])
        feature_importance = metadata_analysis.get('feature_importance', {})
        
        if risk_factors:
            recommendations.append({
                'type': 'METADATA_RISK',
                'priority': 2,
                'title': '📊 Behavioral Risk Factors',
                'description': f'Detected {len(risk_factors)} behavioral risk factors.',
                'risk_factors': risk_factors,
                'top_features': list(feature_importance.keys())[:3] if feature_importance else []
            })
        
        return recommendations
    
    def _generate_image_recommendations(self, image_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on image analysis"""
        recommendations = []
        
        consistency = image_analysis.get('image_text_consistency', {})
        quality_indicators = image_analysis.get('image_quality_indicators', {})
        
        if consistency:
            score = consistency.get('score', 0.8)
            if score < 0.4:
                recommendations.append({
                    'type': 'IMAGE_INCONSISTENCY',
                    'priority': 2,
                    'title': '🖼️ Image-Text Mismatch',
                    'description': f'Low consistency between images and text (score: {score:.3f}).',
                    'reasoning': 'Images may not match the described experience.',
                    'suggested_actions': ['Verify image authenticity', 'Check for stock photos']
                })
        
        if quality_indicators:
            poor_quality = quality_indicators.get('poor_quality_indicators', [])
            if poor_quality:
                recommendations.append({
                    'type': 'IMAGE_QUALITY',
                    'priority': 3,
                    'title': '📷 Image Quality Issues',
                    'description': f'Detected {len(poor_quality)} image quality concerns.',
                    'quality_issues': poor_quality
                })
        
        return recommendations
    
    def _generate_business_recommendations(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate business intelligence recommendations"""
        recommendations = []
        
        # Pattern analysis for business owners
        final_verdict = analysis_result.get('final_verdict', {})
        category = final_verdict.get('category', 'LEGITIMATE')
        
        if category in ['SPAM', 'FAKE_REVIEW', 'RATING_MANIPULATION']:
            recommendations.append({
                'type': 'BUSINESS_INTELLIGENCE',
                'priority': 4,
                'title': '💼 Business Impact Analysis',
                'description': 'Fraudulent reviews detected. Consider impact on business metrics.',
                'suggested_actions': [
                    'Monitor for similar patterns',
                    'Review competitor activity',
                    'Enhance authentic review incentives'
                ]
            })
        
        return recommendations
    
    def generate_batch_recommendations(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate recommendations for a batch of reviews"""
        total_reviews = len(batch_results)
        
        # Aggregate statistics
        categories = {}
        actions = {}
        high_confidence_count = 0
        
        for result in batch_results:
            verdict = result.get('final_verdict', {})
            category = verdict.get('category', 'LEGITIMATE')
            action = verdict.get('action', 'APPROVE')
            confidence = verdict.get('confidence', 0.5)
            
            categories[category] = categories.get(category, 0) + 1
            actions[action] = actions.get(action, 0) + 1
            
            if confidence > self.confidence_thresholds['high']:
                high_confidence_count += 1
        
        # Calculate percentages
        removal_rate = (actions.get('REMOVE', 0) / total_reviews) * 100
        flag_rate = (actions.get('FLAG_FOR_REVIEW', 0) / total_reviews) * 100
        approval_rate = (actions.get('APPROVE', 0) / total_reviews) * 100
        
        return {
            'summary': {
                'total_reviews': total_reviews,
                'removal_rate': removal_rate,
                'flag_rate': flag_rate,
                'approval_rate': approval_rate,
                'high_confidence_rate': (high_confidence_count / total_reviews) * 100
            },
            'category_breakdown': categories,
            'action_breakdown': actions,
            'recommendations': [
                {
                    'type': 'BATCH_SUMMARY',
                    'title': f'📈 Batch Analysis Complete',
                    'description': f'Processed {total_reviews} reviews: {removal_rate:.1f}% flagged for removal, {approval_rate:.1f}% approved.',
                },
                {
                    'type': 'QUALITY_METRICS',
                    'title': '🎯 Quality Assessment',
                    'description': f'High confidence classifications: {high_confidence_count}/{total_reviews} ({(high_confidence_count/total_reviews)*100:.1f}%)',
                }
            ]
        }
