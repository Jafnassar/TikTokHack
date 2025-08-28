#!/usr/bin/env python3
"""
Advanced Review Classification Pipeline using Qwen 2.5 8B
Hackathon Solution for Review Quality & Spam Detection
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix
import json
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class QwenReviewClassifier:
    def __init__(self, model_name="Qwen/Qwen3-8B"):
        """Initialize Qwen-based review classifier"""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Classification categories
        self.categories = {
            'LEGITIMATE': 'A genuine, relevant review about the product/service',
            'SPAM': 'Promotional content, contains contact info, or solicitation',
            'ADVERTISEMENTS': 'Direct advertising content, product promotions, or commercial solicitations',
            'IRRELEVANT': 'Off-topic, not about the product/service being reviewed',
            'FAKE_RANT': 'Emotional outburst without constructive feedback',
            'LOW_QUALITY': 'Very short, generic, or uninformative content'
        }
        
        print(f"🤖 Initializing Qwen Review Classifier")
        print(f"📱 Device: {self.device}")
        print(f"🎯 Categories: {list(self.categories.keys())}")
    
    def load_model(self):
        """Load Qwen model with RTX 4060 optimizations"""
        print(f"🔄 Loading {self.model_name} optimized for RTX 4060...")
        
        # Aggressive quantization for 8GB VRAM
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True  # CPU offload for memory
        )
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                pad_token="<|endoftext|>"
            )
            
            # Load model with aggressive memory optimization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                max_memory={0: "6GB", "cpu": "8GB"}  # Reserve memory for system
            )
            
            print("✅ Qwen 3 8B model loaded successfully on RTX 4060!")
            
        except Exception as e:
            print(f"❌ Error loading Qwen 3 8B: {e}")
            print("💡 Falling back to smaller Qwen model...")
            self._load_fallback_model()
        
        return self
    
    def _load_fallback_model(self):
        """Fallback to smaller Qwen model for 8GB GPU"""
        fallback_model = "Qwen/Qwen2.5-3B-Instruct"  # Much smaller
        print(f"🔄 Loading fallback model: {fallback_model}")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                fallback_model,
                trust_remote_code=True,
                pad_token="<|endoftext|>"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                fallback_model,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            print(f"✅ Fallback model {fallback_model} loaded successfully!")
            print("📝 Note: Using Qwen 2.5 3B with Qwen 3 architecture")
            
        except Exception as e:
            print(f"❌ Fallback model failed: {e}")
            raise e
    
    def create_classification_prompt(self, review_text, include_examples=True):
        """Create a sophisticated prompt for review classification"""
        
        examples = """
Examples:
- "Check out our amazing new product line! 50% off today only!" → ADVERTISEMENTS
- "My yard was a disaster until I called GreenThumb Landscaping. They offered free consultation and mentioned they only have a few spots left for their fall promotion. Call soon!" → ADVERTISEMENTS
- "I've been struggling with weight loss for years. Then I tried NutriSlim supplements and lost 30 pounds in 2 weeks! Use code SAVE20 for 20% off at nutrislim.com" → ADVERTISEMENTS
- "Amazing product! Call 555-1234 for special deals!" → SPAM
- "Best place ever!!! Amazing food amazing service amazing everything!!! 5 stars definitely recommend to everyone!!!" → SPAM
- "This blender is great for smoothies and easy to clean." → LEGITIMATE  
- "Went to this restaurant last night with my family. The food was decent but nothing special. Service was a bit slow but the waiter was friendly." → LEGITIMATE
- "I hate Mondays and traffic jams are terrible." → IRRELEVANT
- "WORST THING EVER!!! HORRIBLE!!!" → FAKE_RANT
- "Good" → LOW_QUALITY
"""
        
        prompt = f"""You are an expert content moderator for an e-commerce platform. Your job is to classify product reviews to maintain platform quality and user trust.

CLASSIFICATION CATEGORIES:
- LEGITIMATE: Genuine, relevant review about the product/service with balanced feedback
- SPAM: Low-quality promotional content, excessive enthusiasm, repetitive text, or generic praise/criticism
- ADVERTISEMENTS: Professional promotional content, business advertising, or commercial solicitations (including sophisticated disguised ads)
- IRRELEVANT: Off-topic, not about the product/service being reviewed  
- FAKE_RANT: Emotional outburst without constructive feedback
- LOW_QUALITY: Very short, generic, or uninformative content

KEY DISTINCTIONS:
- SPAM vs ADVERTISEMENTS: SPAM is typically low-quality repetitive text, while ADVERTISEMENTS are professional promotional content
- ADVERTISEMENTS include: Business names, promotional offers, contact info, professional marketing language
- SPAM includes: Excessive punctuation (!!!), repetitive words, overly enthusiastic generic praise

{examples if include_examples else ""}

REVIEW TO CLASSIFY:
"{review_text}"

Analyze this review considering:
1. Relevance to a product/service
2. Presence of promotional/contact information or business names
3. Commercial advertising or promotional intent (including sophisticated disguised advertisements)
4. Specific company names, promotional offers, urgency language, or calls to action
5. Quality and professionalism vs low-quality repetitive content
6. Constructiveness and specificity
7. Emotional tone vs factual content
8. Length and informativeness
9. Narrative structure that seems too perfect or promotional

CRITICAL RULES:
- If text mentions specific business names + promotional offers → ADVERTISEMENTS
- If text has discount codes, promotional pricing, or contact information → ADVERTISEMENTS
- If text is repetitive, excessive punctuation, generic enthusiasm → SPAM
- If text mentions weight loss products, supplements with claims → ADVERTISEMENTS

Respond with ONLY the category name: LEGITIMATE, SPAM, ADVERTISEMENTS, IRRELEVANT, FAKE_RANT, or LOW_QUALITY"""

        return prompt
    
    def classify_review(self, review_text, max_length=512, temperature=0.6, use_thinking=True):
        """Classify a single review using Qwen with thinking mode (if supported)"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Create prompt
        prompt = self.create_classification_prompt(review_text)
        
        # Check if model supports thinking mode (Qwen 3 feature)
        supports_thinking = hasattr(self.tokenizer, 'apply_chat_template')
        
        if supports_thinking and use_thinking:
            return self._classify_with_thinking(prompt, temperature)
        else:
            return self._classify_standard(prompt, temperature, max_length)
    
    def _classify_with_thinking(self, prompt, temperature=0.6):
        """Classify using Qwen 3 thinking mode"""
        # Prepare messages for Qwen 3 chat format
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Apply chat template with thinking mode
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True  # Qwen 3 thinking capability
            )
            
            # Tokenize
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            # Generate with Qwen 3 optimized parameters
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    temperature=temperature,
                    top_p=0.95,
                    top_k=20,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Extract only the new tokens
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            
            # Parse thinking content and final response
            thinking_content = ""
            final_response = ""
            
            try:
                # Find the </think> token (151668) for Qwen 3
                think_end_index = len(output_ids) - output_ids[::-1].index(151668)
                thinking_content = self.tokenizer.decode(output_ids[:think_end_index], skip_special_tokens=True).strip()
                final_response = self.tokenizer.decode(output_ids[think_end_index:], skip_special_tokens=True).strip()
            except ValueError:
                # No thinking content found, treat as regular response
                final_response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            # Extract category from response
            review_text = prompt.split('"')[1] if '"' in prompt else ""
            category = self.extract_category(final_response, review_text)
            confidence = self.calculate_confidence(review_text, category, thinking_content)
            
            return {
                'category': category,
                'confidence': confidence,
                'thinking_content': thinking_content,
                'final_response': final_response,
                'reasoning': self.get_reasoning(review_text, category, thinking_content)
            }
            
        except Exception as e:
            print(f"⚠️ Thinking mode failed, falling back to standard: {e}")
            return self._classify_standard(prompt, temperature, 512)
    
    def _classify_standard(self, prompt, temperature=0.6, max_length=512):
        """Standard classification for fallback models"""
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # Extract category from response
        review_text = prompt.split('"')[1] if '"' in prompt else prompt
        category = self.extract_category(response, review_text)
        confidence = self.calculate_confidence(review_text, category)
        
        return {
            'category': category,
            'confidence': confidence,
            'thinking_content': "",  # No thinking for fallback
            'final_response': response,
            'reasoning': self.get_reasoning(review_text, category, "")
        }
    
    def extract_category(self, response, review_text=""):
        """Extract category from model response with enhanced content analysis"""
        response_upper = response.upper()
        
        # Try exact matches first
        for category in self.categories.keys():
            if category in response_upper:
                return category
        
        # If no clear category found, use content analysis as fallback
        if review_text:
            content_analysis = self.analyze_content_patterns(review_text)
            ad_score = content_analysis['advertisement_score']
            spam_score = content_analysis['spam_score']
            
            # Use content-based classification
            if ad_score >= 2:  # Strong advertisement indicators
                return 'ADVERTISEMENTS'
            elif spam_score >= 2:  # Strong spam indicators
                return 'SPAM'
            elif len(review_text.split()) < 5:  # Very short
                return 'LOW_QUALITY'
        
        # Enhanced fallback logic with better SPAM vs ADVERTISEMENTS distinction
        # Priority order: ADVERTISEMENTS > SPAM > others
        
        # Strong advertisement indicators (including sophisticated ones)
        advertisement_keywords = [
            'ADVERTISEMENT', 'ADVERTISING', 'PROMOTION', 'PROMOTIONAL', 'SALE', 'DISCOUNT', 
            'DEAL', 'OFFER', 'FREE CONSULTATION', 'LIMITED SPOTS', 'CALL SOON', 'COMPANY NAME',
            'BUSINESS', 'SERVICE', 'WEIGHT LOSS', 'SUPPLEMENT', 'CODE', 'NUTRISLIM',
            'GREENTHUMM', 'LANDSCAPING', 'COMPLIMENTARY', 'PROMO'
        ]
        
        # Spam indicators (low-quality promotional content)
        spam_keywords = [
            'CONTACT', 'PHONE', 'EMAIL', 'CALL NOW', 'AMAZING!!!', 'BEST EVER!!!',
            'REPETITIVE', 'EXCESSIVE', 'GENERIC PRAISE', 'ENTHUSIASTIC'
        ]
        
        if any(word in response_upper for word in advertisement_keywords):
            return 'ADVERTISEMENTS'
        elif any(word in response_upper for word in spam_keywords):
            return 'SPAM'
        elif any(word in response_upper for word in ['IRRELEVANT', 'OFF-TOPIC', 'UNRELATED']):
            return 'IRRELEVANT'
        elif any(word in response_upper for word in ['RANT', 'EMOTIONAL', 'ANGRY']):
            return 'FAKE_RANT'
        elif any(word in response_upper for word in ['SHORT', 'GENERIC', 'LOW']):
            return 'LOW_QUALITY'
        else:
            return 'LEGITIMATE'  # Default fallback
    
    def analyze_content_patterns(self, text):
        """Analyze content patterns to improve classification accuracy"""
        text_lower = text.lower()
        
        # Check for sophisticated advertisement patterns
        ad_patterns = {
            'business_mention': bool(re.search(r'\b[A-Z][a-zA-Z]*\s+[A-Z][a-zA-Z]*\b', text)),  # Company names
            'promotional_offers': any(phrase in text_lower for phrase in [
                'free consultation', 'free service', 'complimentary', 'promotion', 
                'discount', 'sale', 'deal', 'offer', 'code'
            ]),
            'urgency_language': any(phrase in text_lower for phrase in [
                'call soon', 'limited spots', 'limited time', 'before it fills',
                'only have', 'spots left', 'act fast'
            ]),
            'contact_info': bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|@\w+\.\w+', text)),
            'website_mention': bool(re.search(r'\b\w+\.(com|net|org)\b', text_lower))
        }
        
        # Check for spam patterns
        spam_patterns = {
            'excessive_punctuation': text.count('!') > 3 or text.count('?') > 2,
            'repetitive_words': self._has_repetitive_words(text),
            'generic_enthusiasm': any(phrase in text_lower for phrase in [
                'amazing', 'best ever', 'incredible', 'fantastic'
            ]) and len(text.split()) < 20,
            'all_caps': sum(1 for c in text if c.isupper()) / len(text) > 0.3 if text else False
        }
        
        return {
            'advertisement_score': sum(ad_patterns.values()),
            'spam_score': sum(spam_patterns.values()),
            'patterns': {**ad_patterns, **spam_patterns}
        }
    
    def _has_repetitive_words(self, text):
        """Check for repetitive word patterns"""
        words = text.lower().split()
        if len(words) < 5:
            return False
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        return any(count > len(words) * 0.4 for count in word_counts.values())
    
    def calculate_confidence(self, review_text, category, thinking_content=""):
        """Calculate confidence score with Qwen 3 thinking analysis"""
        text_lower = review_text.lower()
        
        # Feature-based confidence calculation
        confidence_factors = []
        
        # Length factor
        length = len(review_text)
        if length > 100:
            confidence_factors.append(0.8)
        elif length > 50:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)
        
        # Thinking content quality factor (Qwen 3 enhancement)
        if thinking_content:
            thinking_words = len(thinking_content.split())
            if thinking_words > 30:
                confidence_factors.append(0.9)  # Detailed thinking
            elif thinking_words > 15:
                confidence_factors.append(0.8)  # Moderate thinking
            elif thinking_words > 5:
                confidence_factors.append(0.7)  # Basic thinking
            
            # Analyze thinking content for certainty indicators
            thinking_lower = thinking_content.lower()
            certainty_indicators = ['clearly', 'obviously', 'definitely', 'certainly', 'unmistakably']
            uncertainty_indicators = ['might', 'could', 'possibly', 'perhaps', 'maybe']
            
            if any(indicator in thinking_lower for indicator in certainty_indicators):
                confidence_factors.append(0.85)
            elif any(indicator in thinking_lower for indicator in uncertainty_indicators):
                confidence_factors.append(0.65)
        
        # Category-specific confidence
        if category == 'SPAM':
            # High confidence if clear spam indicators
            spam_indicators = ['call', 'phone', 'email', 'website', 'discount', 'deal']
            if any(indicator in text_lower for indicator in spam_indicators):
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.6)
                
        elif category == 'ADVERTISEMENTS':
            # High confidence for clear advertising language
            ad_indicators = ['sale', 'discount', 'promotion', 'offer', 'deal', 'buy now', 'limited time']
            sophisticated_ad_indicators = ['free consultation', 'free service', 'call soon', 'limited spots', 'promotion', 'only have', 'spots left', 'complimentary']
            
            if any(indicator in text_lower for indicator in ad_indicators):
                confidence_factors.append(0.85)
            elif any(indicator in text_lower for indicator in sophisticated_ad_indicators):
                confidence_factors.append(0.8)  # High confidence for sophisticated ads
            else:
                confidence_factors.append(0.7)
                
        elif category == 'LOW_QUALITY':
            # High confidence for very short reviews
            if length < 20:
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.7)
        
        elif category == 'LEGITIMATE':
            # High confidence for detailed, relevant content
            if length > 100 and not any(spam in text_lower for spam in ['call', 'email', 'phone']):
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.7)
        
        return min(0.95, np.mean(confidence_factors))
    
    def get_reasoning(self, review_text, category, thinking_content=""):
        """Provide human-readable reasoning enhanced with Qwen 3 thinking analysis"""
        text_lower = review_text.lower()
        length = len(review_text)
        
        # Base reasoning from traditional analysis
        base_reasoning = ""
        if category == 'SPAM':
            if any(contact in text_lower for contact in ['call', 'phone', 'email']):
                base_reasoning = "Contains contact information or promotional solicitation"
            else:
                base_reasoning = "Detected promotional language patterns"
        
        elif category == 'ADVERTISEMENTS':
            if any(ad_word in text_lower for ad_word in ['sale', 'discount', 'promotion', 'offer']):
                base_reasoning = "Contains direct advertising or promotional content"
            else:
                base_reasoning = "Detected commercial advertising language"
        
        elif category == 'IRRELEVANT':
            base_reasoning = "Content appears unrelated to product/service review"
        
        elif category == 'FAKE_RANT':
            base_reasoning = "Excessive emotional language without constructive feedback"
        
        elif category == 'LOW_QUALITY':
            if length < 20:
                base_reasoning = f"Very short review ({length} characters) lacking detail"
            else:
                base_reasoning = "Generic or uninformative content"
        
        elif category == 'LEGITIMATE':
            if length > 100:
                base_reasoning = "Detailed, relevant review with specific feedback"
            else:
                base_reasoning = "Appears to be genuine product/service feedback"
        
        else:
            base_reasoning = "Standard classification based on content analysis"
        
        # Enhance with thinking content if available (Qwen 3 feature)
        if thinking_content and len(thinking_content.strip()) > 10:
            # Extract key insights from thinking
            thinking_lower = thinking_content.lower()
            key_phrases = []
            
            # Look for reasoning indicators in thinking
            if 'because' in thinking_lower:
                key_phrases.append("causal reasoning")
            if 'evidence' in thinking_lower or 'indicates' in thinking_lower:
                key_phrases.append("evidence-based analysis")
            if 'pattern' in thinking_lower:
                key_phrases.append("pattern recognition")
            if 'context' in thinking_lower:
                key_phrases.append("contextual analysis")
            
            if key_phrases:
                enhanced_reasoning = f"{base_reasoning}. Enhanced with: {', '.join(key_phrases)}"
            else:
                enhanced_reasoning = f"{base_reasoning}. Enhanced with deep reasoning analysis"
            
            return enhanced_reasoning
        
        return base_reasoning
    
    def batch_classify(self, reviews_df, batch_size=8):
        """Classify multiple reviews efficiently"""
        print(f"🔄 Processing {len(reviews_df)} reviews in batches of {batch_size}...")
        
        results = []
        total_batches = len(reviews_df) // batch_size + (1 if len(reviews_df) % batch_size else 0)
        
        for i in range(0, len(reviews_df), batch_size):
            batch = reviews_df.iloc[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            print(f"📊 Processing batch {batch_num}/{total_batches}")
            
            for _, row in batch.iterrows():
                try:
                    result = self.classify_review(row['text'])
                    result.update({
                        'review_id': row.get('id', i),
                        'original_text': row['text'],
                        'original_rating': row.get('rating', None)
                    })
                    results.append(result)
                    
                except Exception as e:
                    print(f"⚠️ Error processing review: {e}")
                    results.append({
                        'review_id': row.get('id', i),
                        'original_text': row['text'],
                        'category': 'LEGITIMATE',  # Safe fallback
                        'confidence': 0.5,
                        'error': str(e)
                    })
        
        return pd.DataFrame(results)
    
    def enforce_policy(self, classification_result):
        """Enforce content policy based on classification"""
        category = classification_result['category']
        confidence = classification_result['confidence']
        
        if category == 'SPAM' and confidence > 0.7:
            return {
                'action': 'REMOVE',
                'reason': 'Spam content detected with high confidence',
                'escalate': False
            }
        
        elif category == 'ADVERTISEMENTS' and confidence > 0.7:
            return {
                'action': 'REMOVE',
                'reason': 'Advertisement content detected - commercial solicitation',
                'escalate': False
            }
        
        elif category == 'IRRELEVANT' and confidence > 0.8:
            return {
                'action': 'REMOVE',
                'reason': 'Content not relevant to product/service',
                'escalate': False
            }
        
        elif category in ['FAKE_RANT', 'LOW_QUALITY'] and confidence > 0.8:
            return {
                'action': 'FLAG_FOR_REVIEW',
                'reason': f'{category} content detected',
                'escalate': True
            }
        
        elif category == 'LEGITIMATE':
            return {
                'action': 'APPROVE',
                'reason': 'Content appears legitimate',
                'escalate': False
            }
        
        else:
            return {
                'action': 'FLAG_FOR_REVIEW',
                'reason': f'Uncertain classification: {category} (confidence: {confidence:.2f})',
                'escalate': True
            }
    
    def generate_report(self, results_df):
        """Generate comprehensive classification report"""
        print("\n" + "="*60)
        print("📊 QWEN REVIEW CLASSIFICATION REPORT")
        print("="*60)
        
        # Overall statistics
        total_reviews = len(results_df)
        avg_confidence = results_df['confidence'].mean()
        
        print(f"📈 Total Reviews Processed: {total_reviews}")
        print(f"🎯 Average Confidence: {avg_confidence:.3f}")
        
        # Category distribution
        print(f"\n📋 Category Distribution:")
        category_counts = results_df['category'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / total_reviews) * 100
            print(f"   {category}: {count} ({percentage:.1f}%)")
        
        # Policy enforcement
        print(f"\n⚖️ Policy Enforcement:")
        policy_results = results_df.apply(lambda row: self.enforce_policy(row), axis=1)
        actions = pd.DataFrame(policy_results.tolist())['action'].value_counts()
        for action, count in actions.items():
            percentage = (count / total_reviews) * 100
            print(f"   {action}: {count} ({percentage:.1f}%)")
        
        # High confidence classifications
        high_conf = results_df[results_df['confidence'] > 0.8]
        print(f"\n🔥 High Confidence Classifications (>0.8): {len(high_conf)} ({len(high_conf)/total_reviews*100:.1f}%)")
        
        return {
            'total_processed': total_reviews,
            'average_confidence': avg_confidence,
            'category_distribution': category_counts.to_dict(),
            'high_confidence_count': len(high_conf)
        }


def main():
    """Demo the Qwen Review Classification Pipeline"""
    print("🚀 Starting Qwen Review Classification Pipeline Demo")
    
    # Initialize classifier
    classifier = QwenReviewClassifier()
    classifier.load_model()
    
    # Test with sample reviews
    test_reviews = [
        "This product is amazing! Great quality and fast shipping. Highly recommend!",
        "Call 555-1234 for special discount on bulk orders! Best deals in town!",
        "I hate Mondays and my boss is annoying. Traffic was terrible today.",
        "WORST PRODUCT EVER!!! TERRIBLE!!! WASTE OF MONEY!!!",
        "Good",
        "The build quality exceeded my expectations. The materials feel premium and the design is both functional and aesthetically pleasing. After using it for 3 weeks, I can confidently say this is worth the investment."
    ]
    
    print(f"\n🧪 Testing with {len(test_reviews)} sample reviews...")
    
    # Create test dataframe
    test_df = pd.DataFrame({
        'id': range(len(test_reviews)),
        'text': test_reviews,
        'rating': [5, 5, 1, 1, 3, 5]
    })
    
    # Classify reviews
    results = classifier.batch_classify(test_df)
    
    # Generate report
    report = classifier.generate_report(results)
    
    # Show detailed results
    print(f"\n🔍 Detailed Results:")
    for _, result in results.iterrows():
        print(f"\n📝 Review: {result['original_text'][:60]}...")
        print(f"📊 Category: {result['category']} (confidence: {result['confidence']:.2f})")
        print(f"💭 Reasoning: {result['reasoning']}")
        
        policy = classifier.enforce_policy(result)
        print(f"⚖️ Policy Action: {policy['action']} - {policy['reason']}")
    
    print(f"\n✅ Pipeline demonstration completed!")
    return classifier, results

if __name__ == "__main__":
    classifier, results = main()
