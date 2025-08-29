"""
Quick validation test for the finalized app
"""
import sys
import os
sys.path.append('backend')

def test_imports():
    """Test that all imports work correctly"""
    try:
        from backend.models.qwen_classifier import QwenClassifier
        from backend.models.metadata_classifier import MetadataClassifier  
        from backend.utils.feature_extractor import FeatureExtractor
        from backend.utils.recommendation_engine import RecommendationEngine
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without loading models"""
    try:
        from backend.utils.feature_extractor import FeatureExtractor
        from backend.utils.recommendation_engine import RecommendationEngine
        
        # Test feature extraction
        extractor = FeatureExtractor()
        features = extractor.extract_features("This is a test review", rating=5)
        print(f"✅ Feature extraction works: {len(features)} features extracted")
        
        # Test recommendation engine with mock data
        engine = RecommendationEngine()
        mock_analysis = {
            'qwen_analysis': {'label': 'fake', 'confidence': 0.8},
            'metadata_analysis': {'label': 'fake', 'confidence': 0.7},
            'features': features,
            'final_verdict': {'label': 'fake', 'confidence': 0.8, 'action': 'REMOVE'}
        }
        recommendations = engine.generate_recommendations(mock_analysis)
        print(f"✅ Recommendation engine works: {len(recommendations)} recommendations generated")
        
        return True
    except Exception as e:
        print(f"❌ Functionality error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Running quick validation test...")
    print("=" * 50)
    
    success = True
    success &= test_imports()
    success &= test_basic_functionality()
    
    print("=" * 50)
    if success:
        print("🎉 All tests passed! App should be working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
