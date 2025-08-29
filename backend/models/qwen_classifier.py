#!/usr/bin/env python3
"""
Qwen LLM Model Handler
Handles all Qwen 2.5-3B-Instruct model operations with training data integration
"""

import torch
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False

class QwenClassifier:
    def __init__(self):
        """Initialize Qwen 2.5-3B model with optimized configuration"""
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Classification categories
        self.categories = {
            'LEGITIMATE': {'priority': 1, 'action': 'APPROVE', 'color': '🟢'},
            'SPAM': {'priority': 2, 'action': 'REMOVE', 'color': '🔴'},
            'FAKE_REVIEW': {'priority': 3, 'action': 'REMOVE', 'color': '🔴'},
            'NO_EXPERIENCE': {'priority': 4, 'action': 'REMOVE', 'color': '🔴'},
            'RATING_MANIPULATION': {'priority': 5, 'action': 'REMOVE', 'color': '🔴'},
            'REPETITIVE_SPAM': {'priority': 6, 'action': 'REMOVE', 'color': '🔴'},
            'LOCATION_MISMATCH': {'priority': 7, 'action': 'FLAG_FOR_REVIEW', 'color': '🟡'},
            'SUSPICIOUS_USER_PATTERN': {'priority': 8, 'action': 'FLAG_FOR_REVIEW', 'color': '🟡'},
            'IMAGE_TEXT_MISMATCH': {'priority': 9, 'action': 'FLAG_FOR_REVIEW', 'color': '🟡'},
            'TEMPORAL_ANOMALY': {'priority': 10, 'action': 'FLAG_FOR_REVIEW', 'color': '🟡'}
        }
        
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the Qwen model with optimized configuration"""
        if not QWEN_AVAILABLE:
            print("⚠️ Transformers library not available for Qwen model")
            return
        
        try:
            print("🤖 Loading Qwen 2.5-3B-Instruct model...")
            
            # Configure quantization for efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            # Load model and tokenizer
            model_name = "Qwen/Qwen2.5-3B-Instruct"
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Try loading with quantization first, fallback to regular loading
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    quantization_config=quantization_config,
                    trust_remote_code=True
                )
                print("✅ Qwen model loaded with 4-bit quantization")
            except Exception as e:
                print(f"⚠️ Quantization failed, loading without: {e}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                print("✅ Qwen model loaded successfully")
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            print(f"❌ Error loading Qwen model: {e}")
            self.model = None
            self.tokenizer = None
    
    def classify_review(self, review_text, rating=None, user_metadata=None, business_name=None):
        """Classify review using Qwen LLM with training data context"""
        if not self.model or not self.tokenizer:
            return {
                'category': 'LEGITIMATE',
                'confidence': 0.5,
                'reasoning': 'Qwen model not available',
                'method': 'fallback'
            }
        
        try:
            # Create enhanced prompt with training data context
            prompt = self._create_training_enhanced_prompt(
                review_text, rating, user_metadata, business_name
            )
            
            # Tokenize and generate
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
                padding=True
            )
            
            # Move to device
            device = self.model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(prompt):].strip()
            
            # Parse the structured response
            return self._parse_qwen_response(response)
            
        except Exception as e:
            print(f"⚠️ Error in Qwen classification: {e}")
            return {
                'category': 'LEGITIMATE',
                'confidence': 0.5,
                'reasoning': f'Error in Qwen analysis: {str(e)}',
                'method': 'error_fallback'
            }
    
    def _create_training_enhanced_prompt(self, review_text, rating, user_metadata, business_name):
        """Create enhanced prompt with comprehensive training data insights"""
        
        # Get user behavior context
        user_context = ""
        if user_metadata:
            if isinstance(user_metadata, dict):
                review_count = user_metadata.get('review_count', 1)
                avg_rating = user_metadata.get('avg_rating', 3.0)
                rating_std = user_metadata.get('rating_std', 0.0)
                
                user_context = f"""
User Profile Context:
- Total reviews written: {review_count}
- Average rating given: {avg_rating:.1f}
- Rating consistency (std dev): {rating_std:.2f}
- User type: {'Prolific reviewer' if review_count > 50 else 'Casual reviewer' if review_count > 10 else 'New user'}
"""
            else:
                user_context = f"User Context: {user_metadata}"
        
        # Get business context
        business_context = f"Business: {business_name}" if business_name else "Business: Unknown"
        
        # Rating context with mismatch detection
        rating_context = ""
        if rating:
            rating_context = f"Given Rating: {rating} stars"
            # Add sentiment analysis hint
            negative_words = ['terrible', 'awful', 'horrible', 'worst', 'bad', 'cold', 'rude', 'disappointing']
            positive_words = ['amazing', 'excellent', 'perfect', 'best', 'great', 'fantastic', 'wonderful']
            
            review_lower = review_text.lower()
            negative_count = sum(1 for word in negative_words if word in review_lower)
            positive_count = sum(1 for word in positive_words if word in review_lower)
            
            if rating >= 4 and negative_count > positive_count:
                rating_context += f" ⚠️ RATING MISMATCH: {rating}-star rating with negative content"
            elif rating <= 2 and positive_count > negative_count:
                rating_context += f" ⚠️ RATING MISMATCH: {rating}-star rating with positive content"
        else:
            rating_context = "Rating: Not provided"
        
        prompt = f"""You are an expert review authenticity classifier. Analyze this review for fake patterns.

🚨 FAKE REVIEW DETECTION CHECKLIST:

IMMEDIATE SPAM INDICATORS (classify as SPAM):
• Phone numbers or contact info: (555) 123-XXXX, "text us", "call now"
• Booking instructions: "book now", "schedule appointment", "get on list"
• Marketing packages: quoted package names, specific pricing
• Professional marketing language in casual reviews

FAKE CONTENT INDICATORS (classify as FAKE_REVIEW):
• Overly perfect experiences with no flaws
• Generic praise that could apply anywhere
• Professional marketing terminology 
• Claims that seem impossible or unrealistic
• Template-like language patterns

RATING MANIPULATION (classify as RATING_MANIPULATION):
• Rating doesn't match review content sentiment
• Extreme rating (1 or 5) with mild content
• 5-star rating with negative words: terrible, awful, horrible, cold, bad, disappointing
• 1-star rating with positive words: amazing, excellent, perfect, great, fantastic

NO EXPERIENCE (classify as NO_EXPERIENCE):
• Admits not visiting: "haven't been", "planning to visit"
• Based on hearsay: "I heard", "my friend said"

LEGITIMATE REVIEWS have:
• Specific personal details and honest opinions
• Mix of positive and negative aspects
• Natural conversational tone
• Realistic experience descriptions

REVIEW TO ANALYZE:
{user_context}
{business_context}
{rating_context}

Review Text: "{review_text}"

Analyze this review and respond EXACTLY in this format:
CATEGORY: [LEGITIMATE, SPAM, FAKE_REVIEW, NO_EXPERIENCE, RATING_MANIPULATION, REPETITIVE_SPAM, SUSPICIOUS_USER_PATTERN, LOCATION_MISMATCH, IMAGE_TEXT_MISMATCH, or TEMPORAL_ANOMALY]
CONFIDENCE: [0.000-1.000]
REASONING: [Brief explanation of why you classified it this way]

Analysis:"""
        
        return prompt
    
    def _parse_qwen_response(self, response):
        """Parse Qwen's structured response"""
        try:
            lines = response.strip().split('\n')
            category = 'LEGITIMATE'
            confidence = 0.5
            reasoning = 'Unable to parse response'
            
            for line in lines:
                line = line.strip()
                if line.startswith('CATEGORY:'):
                    category = line.split(':', 1)[1].strip()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                    except:
                        confidence = 0.5
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()
            
            # Validate category
            if category not in self.categories:
                category = 'LEGITIMATE'
            
            # Clamp confidence
            confidence = max(0.0, min(1.0, confidence))
            
            return {
                'category': category,
                'confidence': confidence,
                'reasoning': reasoning,
                'method': 'qwen_llm',
                'raw_response': response
            }
            
        except Exception as e:
            return {
                'category': 'LEGITIMATE',
                'confidence': 0.5,
                'reasoning': f'Error parsing response: {str(e)}',
                'method': 'parse_error'
            }
    
    def is_available(self):
        """Check if Qwen model is available"""
        return self.model is not None and self.tokenizer is not None
