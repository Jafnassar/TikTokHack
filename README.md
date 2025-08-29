# 🎭 TikTok Review Analysis Dashboard

Modern AI-powered fake review detection system using Qwen 2.5-3B-Instruct transformer model with multimodal classification and clean, professional interface.

## 🚀 Features

- **🤖 Advanced AI Classification**: Real Qwen transformer model with contextual understanding
- **🎨 Modern Dashboard**: Clean, professional interface with Inter font and gradient design
- **🔍 Multimodal Analysis**: Combines text analysis, metadata patterns, and feature extraction
- **⭐ Rating-Text Contradiction Detection**: Intelligent mismatch identification
- **📊 Real-time Metrics**: Interactive dashboard with hover effects and responsive design
- **🎯 Smart Recommendations**: AI-powered action suggestions with confidence scoring
- **⚡ Optimized Performance**: 4-bit quantization for efficient GPU usage
- **🧪 Comprehensive Testing**: Full test suite for all classification components

## 🏃‍♂️ Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

3. **Access Dashboard**: Open http://localhost:8501 in your browser

## 📁 Project Structure

```
TikTokHack/
├── app.py                          # 🚀 Main application entry point
├── README.md                       # 📖 Project documentation
├── requirements.txt                # 📦 Python dependencies
├── backend/                        # 🔧 Core AI models and utilities
│   ├── models/
│   │   ├── qwen_classifier.py     # 🤖 Qwen AI classifier
│   │   └── metadata_classifier.py # 📊 Metadata analysis
│   └── utils/
│       ├── feature_extractor.py   # 🔍 Feature extraction
│       └── recommendation_engine.py # 🎯 Smart recommendations
├── frontend/                       # 🎨 UI components
│   └── components/
│       └── dashboard_components.py # 📊 Dashboard widgets
├── tests/                          # 🧪 Test suite
│   ├── test_*.py                  # Unit tests
│   ├── debug_*.py                 # Debug utilities
│   └── *_test.py                  # Integration tests
└── data/                          # 📂 Sample datasets
    └── Google Map Reviews/
```

## 🔧 Architecture

### Core Components

1. **QwenReviewClassifier**: 4-bit quantized Qwen 2.5-3B model for text classification
2. **MetadataClassifier**: Pattern-based analysis for review metadata
3. **FeatureExtractor**: Multi-dimensional feature extraction from review text
4. **RecommendationEngine**: AI-powered action recommendations
5. **DashboardComponents**: Modern UI components with professional styling

### AI Pipeline

```
Review Text → Feature Extraction → Qwen Classification → Metadata Analysis → Recommendation → Display
```

## 📊 Classification Categories

### High Priority (Remove/Flag)
- **SPAM**: Contact info, promotional content, solicitation
- **ADVERTISEMENTS**: Professional marketing, business promotions  
- **FAKE**: Obviously fabricated or false content
- **OFFENSIVE**: Inappropriate language, personal attacks
- **REPETITIVE_SPAM**: Auto-generated content with repeated phrases
- **NO_EXPERIENCE**: User admits never using product/service
- **RATING_TEXT_MISMATCH**: Rating conflicts with text sentiment
- **COMPETITOR_COMPARISON**: Primarily compares to other businesses
- **IRRELEVANT**: Off-topic, not about the product/service
- **LOW_QUALITY**: Very short, generic, uninformative

### Legitimate Categories (Approve)
- **LEGITIMATE**: Balanced, specific feedback about product/service
- **SERVICE_FOCUSED**: About staff quality, customer service, interaction
- **VALUE_FOCUSED**: About pricing, value for money, cost considerations
- **CLEANLINESS_FOCUSED**: About hygiene, sanitation, cleanliness standards  
- **WAIT_TIME_FOCUSED**: About service speed, waiting times, efficiency
- **LOCATION_FOCUSED**: About accessibility, parking, location convenience
- **PORTION_SIZE_FOCUSED**: About food quantity, portion satisfaction

### Language
- **NON_ENGLISH**: Review written in non-English language

## 🛠️ Quick Start

### 1. Install Dependencies
```bash
pip install -r qwen_requirements.txt
```

### 2. Run Dashboard
```bash
streamlit run qwen_dashboard.py
```

### 3. Load Model
- Click "🚀 Load Qwen LLM Model" in sidebar
- Wait 2-3 minutes for initial model loading
- Start classifying reviews

## 💻 System Requirements

- **GPU**: RTX 4060 or equivalent (8GB+ VRAM recommended)
- **RAM**: 16GB+ system memory
- **Python**: 3.8+
- **CUDA**: Compatible CUDA installation

## 🎯 Usage Examples

### Single Review with Rating Analysis
```python
from qwen_review_pipeline import QwenReviewClassifier

classifier = QwenReviewClassifier()
classifier.load_model()

result = classifier.classify_review(
    "The service was excellent and the food was delicious!", 
    rating=5  # Include star rating for mismatch detection
)

print(f"Category: {result['category']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Action: {result['action']}")
print(f"Rating Analysis: {result['pattern_analysis']['pattern_scores']['mismatch_indicators']}")
```

### Batch Processing with Ratings
Upload CSV with columns:
- `review` or `text`: Review content
- `rating` or `stars`: Star ratings (1-5)

Dashboard automatically detects:
- Rating-text mismatches
- Feature scores for all reviews
- Comprehensive batch statistics

### Rating Contradiction Examples
```python
# High rating but negative sentiment
result = classifier.classify_review(
    "Terrible food, worst service ever, complete waste of money!",
    rating=5
)
# → Category: RATING_TEXT_MISMATCH

# Low rating but positive sentiment  
result = classifier.classify_review(
    "Amazing food, incredible service, highly recommend!",
    rating=1
)
# → Category: RATING_TEXT_MISMATCH
```

## 📈 Enhanced Features

### Pattern Analysis
- 15+ pattern detection algorithms
- Pre-classification guidance for LLM
- Keyword/phrase pattern matching
- Repetitive content detection
- Language detection

### LLM Integration  
- Qwen 2.5-3B-Instruct transformer
- Thinking mode for complex reasoning
- Context-aware classification
- Sentiment-rating alignment analysis

### Confidence Scoring
- Hybrid scoring: 70% LLM + 30% pattern agreement
- Enhanced reasoning generation
- Priority-based category selection

## 🔧 Technical Architecture

```
Enhanced Classification Pipeline:
├── Input: Review text + optional rating
├── Pattern Analysis: 15+ detection algorithms  
├── Enhanced Prompt: Pattern guidance + examples
├── LLM Processing: Qwen thinking mode
├── Result Enhancement: Confidence + reasoning
└── Output: Category + action + detailed analysis
```

## 📝 Files

- **`qwen_dashboard.py`** - Streamlit web interface
- **`qwen_review_pipeline.py`** - Core classifier with enhanced patterns
- **`qwen_requirements.txt`** - Python dependencies
- **`reviews_cleaned.csv`** - Training/test dataset

## ⚡ Performance

- **Accuracy**: Enhanced through pattern + LLM combination
- **Speed**: 30-60 seconds per review (real AI inference)
- **Memory**: ~4GB GPU VRAM with 4-bit quantization
- **Throughput**: 1-2 reviews per minute (quality over speed)

## 🎪 TikTok Hackathon Solution

This enhanced classifier provides:
- Real AI understanding vs simple pattern matching
- 17 comprehensive categories for complete review management
- Production-ready accuracy and confidence scoring
- Scalable architecture for review platform integration
