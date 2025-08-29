#!/usr/bin/env python3
"""
Qwen LLM Model Handler
Handles all Qwen 2.5-3B-Instruct model operations with training data integration
"""

import torch
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import lru_cache
import re

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
        self.model_loaded = False  # Add missing attribute
        
        # Enhanced classification categories with detailed fake review subcategories
        self.categories = {
            # Legitimate Reviews
            'LEGITIMATE': {'priority': 1, 'action': 'APPROVE', 'color': '🟢'},
            
            # Fake Review Subcategories - Marketing/Commercial
            'ADVERTISEMENT': {'priority': 2, 'action': 'REMOVE', 'color': '🔴'},
            'PROMOTIONAL_CONTENT': {'priority': 3, 'action': 'REMOVE', 'color': '🔴'},
            'BUSINESS_OWNER_FAKE': {'priority': 4, 'action': 'REMOVE', 'color': '🔴'},
            'COMPETITOR_ATTACK': {'priority': 5, 'action': 'REMOVE', 'color': '🔴'},
            
            # Fake Review Subcategories - Content Quality Issues  
            'TOO_GENERIC': {'priority': 6, 'action': 'REMOVE', 'color': '🔴'},
            'TOO_LITTLE_DETAIL': {'priority': 7, 'action': 'REMOVE', 'color': '🔴'},
            'TEMPLATE_LANGUAGE': {'priority': 8, 'action': 'REMOVE', 'color': '🔴'},
            'UNREALISTIC_PRAISE': {'priority': 9, 'action': 'REMOVE', 'color': '🔴'},
            'COPY_PASTE_REVIEW': {'priority': 10, 'action': 'REMOVE', 'color': '🔴'},
            
            # Fake Review Subcategories - Experience Issues
            'NO_ACTUAL_EXPERIENCE': {'priority': 11, 'action': 'REMOVE', 'color': '🔴'},
            'HEARSAY_REVIEW': {'priority': 12, 'action': 'REMOVE', 'color': '🔴'},
            'PLANNING_TO_VISIT': {'priority': 13, 'action': 'REMOVE', 'color': '🔴'},
            'LOCATION_IMPOSSIBLE': {'priority': 14, 'action': 'REMOVE', 'color': '🔴'},
            
            # Fake Review Subcategories - Rating Manipulation
            'RATING_TEXT_MISMATCH': {'priority': 15, 'action': 'REMOVE', 'color': '🔴'},
            'EXTREME_RATING_ABUSE': {'priority': 16, 'action': 'REMOVE', 'color': '🔴'},
            'SENTIMENT_CONTRADICTION': {'priority': 17, 'action': 'REMOVE', 'color': '🔴'},
            
            # Fake Review Subcategories - Spam Patterns
            'REPETITIVE_SPAM': {'priority': 18, 'action': 'REMOVE', 'color': '🔴'},
            'KEYWORD_STUFFING': {'priority': 19, 'action': 'REMOVE', 'color': '🔴'},
            'CONTACT_INFO_SPAM': {'priority': 20, 'action': 'REMOVE', 'color': '🔴'},
            'LINK_SPAM': {'priority': 21, 'action': 'REMOVE', 'color': '🔴'},
            
            # Fake Review Subcategories - Behavioral Issues
            'BOT_GENERATED': {'priority': 22, 'action': 'REMOVE', 'color': '🔴'},
            'MASS_GENERATED': {'priority': 23, 'action': 'REMOVE', 'color': '🔴'},
            'INCENTIVIZED_FAKE': {'priority': 24, 'action': 'REMOVE', 'color': '🔴'},
            'PAID_REVIEW_SERVICE': {'priority': 25, 'action': 'REMOVE', 'color': '🔴'},
            
            # Fake Review Subcategories - Content Problems
            'UNHELPFUL_CONTENT': {'priority': 26, 'action': 'REMOVE', 'color': '🔴'},
            'IRRELEVANT_CONTENT': {'priority': 27, 'action': 'REMOVE', 'color': '🔴'},
            'NONSENSICAL_TEXT': {'priority': 28, 'action': 'REMOVE', 'color': '🔴'},
            'WRONG_BUSINESS': {'priority': 29, 'action': 'REMOVE', 'color': '�'},
            
            # Suspicious but need review
            'SUSPICIOUS_USER_PATTERN': {'priority': 30, 'action': 'FLAG_FOR_REVIEW', 'color': '🟡'},
            'IMAGE_TEXT_MISMATCH': {'priority': 31, 'action': 'FLAG_FOR_REVIEW', 'color': '🟡'},
            'TEMPORAL_ANOMALY': {'priority': 32, 'action': 'FLAG_FOR_REVIEW', 'color': '🟡'},
            'UNUSUAL_LANGUAGE_PATTERN': {'priority': 33, 'action': 'FLAG_FOR_REVIEW', 'color': '🟡'}
        }
        
        # Initialize model
        self.initialize_model()
    
    def classify_batch(self, reviews_data, progress_callback=None):
        """Optimized batch classification with parallel processing"""
        if not self.model_loaded:
            print("⚠️ Model not loaded, falling back to rule-based classification")
            return self._classify_batch_fallback(reviews_data, progress_callback)
        
        results = []
        total_reviews = len(reviews_data)
        
        # Process in smaller batches to optimize memory usage
        batch_size = 4  # Reduced for better memory management
        start_time = time.time()
        
        for i in range(0, total_reviews, batch_size):
            batch = reviews_data[i:i + batch_size]
            batch_results = []
            
            # Process each item in the batch
            for j, review_data in enumerate(batch):
                try:
                    result = self.classify_review(
                        review_data['review_text'],
                        review_data.get('rating'),
                        review_data.get('user_metadata'),
                        review_data.get('business_name')
                    )
                    batch_results.append(result)
                    
                    # Progress callback
                    if progress_callback:
                        progress = (i + j + 1) / total_reviews
                        elapsed = time.time() - start_time
                        eta = (elapsed / (i + j + 1)) * (total_reviews - i - j - 1) if i + j > 0 else 0
                        progress_callback(progress, eta)
                        
                except Exception as e:
                    print(f"⚠️ Error processing review {i + j + 1}: {e}")
                    batch_results.append({
                        'category': 'LEGITIMATE',
                        'confidence': 0.5,
                        'reasoning': f'Error in processing: {str(e)}',
                        'method': 'error_fallback'
                    })
            
            results.extend(batch_results)
            
            # Small delay to prevent overheating
            if i + batch_size < total_reviews:
                time.sleep(0.1)
        
        return results
    
    def _classify_batch_fallback(self, reviews_data, progress_callback=None):
        """Fast rule-based classification for fallback"""
        results = []
        
        for i, review_data in enumerate(reviews_data):
            result = self._fast_rule_based_classification(
                review_data['review_text'],
                review_data.get('rating', 3)
            )
            results.append(result)
            
            if progress_callback:
                progress_callback((i + 1) / len(reviews_data), 0)
        
        return results
    
    @lru_cache(maxsize=128)
    def _fast_rule_based_classification(self, review_text, rating):
        """Fast rule-based classification with caching"""
        template_patterns = self._detect_template_patterns(review_text)
        
        # Quick contact spam detection (most specific first)
        if self._quick_spam_check(review_text):
            return {
                'category': 'CONTACT_INFO_SPAM',
                'confidence': 0.95,
                'reasoning': 'Contact information or promotional content detected',
                'method': 'rule_based_fast'
            }
        
        # Quick no actual experience detection
        no_experience_phrases = [
            "haven't been", "never been", "haven't visited", "haven't actually been",
            "heard it's", "my friend said", "someone told me", "i heard"
        ]
        text_lower = review_text.lower()
        if any(phrase in text_lower for phrase in no_experience_phrases):
            return {
                'category': 'NO_ACTUAL_EXPERIENCE',
                'confidence': 0.9,
                'reasoning': 'Reviewer admits no actual experience or relies on hearsay',
                'method': 'rule_based_fast'
            }
        
        # Quick rating mismatch detection (more precise)
        rating_mismatch = self._quick_rating_mismatch(review_text, rating)
        if rating_mismatch:
            return {
                'category': 'RATING_TEXT_MISMATCH',
                'confidence': 0.85,
                'reasoning': 'Rating does not match review sentiment - possible manipulation',
                'method': 'rule_based_fast'
            }
        
        # Template detection with enhanced patterns
        if template_patterns:
            return {
                'category': 'TEMPLATE_LANGUAGE',
                'confidence': 0.9,
                'reasoning': f'Template patterns detected: {", ".join(template_patterns[:2])}',
                'method': 'rule_based_fast'
            }
        
        # Too generic detection
        if self._is_too_generic(review_text):
            return {
                'category': 'TOO_GENERIC',
                'confidence': 0.8,
                'reasoning': 'Review is too vague and generic to be helpful',
                'method': 'rule_based_fast'
            }
        
        # Too little detail detection
        if len(review_text.split()) <= 4:
            return {
                'category': 'TOO_LITTLE_DETAIL',
                'confidence': 0.8,
                'reasoning': 'Review is extremely short with insufficient detail',
                'method': 'rule_based_fast'
            }
        
        return {
            'category': 'LEGITIMATE',
            'confidence': 0.7,
            'reasoning': 'No obvious fake patterns detected',
            'method': 'rule_based_fast'
        }
    
    def _quick_spam_check(self, text):
        """Quick spam pattern detection"""
        spam_patterns = [
            r'call\s+\d{3}[-.]?\d{3}[-.]?\d{4}',  # Phone numbers
            r'contact\s+us\s+at\s+\d',            # Contact us at number
            r'visit\s+our\s+website',             # Website promotion
            r'check\s+out\s+our',                 # Check out our
            r'@\w+',                              # Social media handles
            r'www\.',                             # Website URLs
            r'\.com',                             # Domain names
            r'catering\s+services',               # Service promotion
            r'for\s+more\s+info'                  # More info requests
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in spam_patterns)
    
    def _is_too_generic(self, text):
        """Detect generic, unhelpful reviews"""
        generic_phrases = [
            "great place", "good place", "nice place", "okay place",
            "recommended", "worth it", "not bad", "pretty good",
            "decent", "fine", "alright", "good service",
            "great service", "good food", "nice food"
        ]
        
        text_lower = text.lower().strip()
        
        # Very short generic phrases
        if any(phrase == text_lower for phrase in generic_phrases):
            return True
        
        # Generic + minimal words
        word_count = len(text.split())
        if word_count <= 6 and any(phrase in text_lower for phrase in generic_phrases):
            return True
        
        return False
    
    def _quick_rating_mismatch(self, text, rating):
        """Enhanced rating-text mismatch detection"""
        if not rating:
            return False
            
        # Stronger negative/positive word lists
        strong_negative = ['terrible', 'awful', 'horrible', 'worst', 'disgusting', 'hate', 'never again', 'terrible', 'cold food', 'rude staff']
        strong_positive = ['amazing', 'excellent', 'perfect', 'best', 'love', 'incredible', 'fantastic', 'outstanding', 'wonderful']
        
        # Moderate negative/positive words
        moderate_negative = ['bad', 'poor', 'disappointing', 'overpriced', 'mediocre', 'slow']
        moderate_positive = ['good', 'great', 'nice', 'friendly', 'delicious', 'tasty']
        
        text_lower = text.lower()
        
        # Count sentiment indicators
        strong_neg_count = sum(1 for word in strong_negative if word in text_lower)
        strong_pos_count = sum(1 for word in strong_positive if word in text_lower)
        moderate_neg_count = sum(1 for word in moderate_negative if word in text_lower)
        moderate_pos_count = sum(1 for word in moderate_positive if word in text_lower)
        
        # Strong mismatches
        if rating >= 4 and strong_neg_count >= 1:  # High rating with strong negative words
            return True
        if rating <= 2 and strong_pos_count >= 1:  # Low rating with strong positive words
            return True
            
        # Moderate mismatches with additional context
        if rating == 5 and moderate_neg_count >= 2 and moderate_pos_count == 0:
            return True
        if rating == 1 and moderate_pos_count >= 2 and moderate_neg_count == 0:
            return True
            
        return False
    
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
            
            # Set model loaded flag
            self.model_loaded = True
                
        except Exception as e:
            print(f"❌ Error loading Qwen model: {e}")
            self.model = None
            self.tokenizer = None
            self.model_loaded = False
    
    def classify_review(self, review_text, rating=None, user_metadata=None, business_name=None):
        """Classify review using Qwen LLM with training data context"""
        if not self.model or not self.tokenizer:
            print("⚠️ Model not available, using fallback classification")
            return self._fast_rule_based_classification(review_text, rating or 3)
        
        try:
            print(f"🔍 Classifying review: {review_text[:50]}...")
            
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
            
            # Generate response with optimized parameters
            try:
                with torch.no_grad():
                    print("🤖 Generating response...")
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=200,     # Increased for complete reasoning
                        do_sample=False,        # Greedy decoding
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1, # Prevent loops
                        early_stopping=True     # Stop when EOS is generated
                    )
                    print("✅ Response generated")
            except Exception as gen_error:
                print(f"⚠️ Generation failed: {gen_error}")
                # Fall back to rule-based classification
                return self._fast_rule_based_classification(review_text, rating or 3)
            
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
    
    def _detect_template_patterns(self, review_text):
        """Detect template language patterns for enhanced classification"""
        text_upper = review_text.upper()
        
        # Template indicators
        template_indicators = []
        
        # Excessive exclamation marks
        exclamation_count = review_text.count('!')
        if exclamation_count >= 4:  # Lowered threshold
            template_indicators.append(f"Excessive exclamation marks ({exclamation_count})")
        
        # All caps words (more sensitive detection)
        words = review_text.split()
        caps_words = [word.strip('!.,?') for word in words if word.strip('!.,?').isupper() and len(word.strip('!.,?')) > 2]
        if len(caps_words) >= 2:  # Lowered threshold
            template_indicators.append(f"Multiple ALL CAPS words: {', '.join(caps_words[:5])}")
        
        # Generic superlatives (expanded list)
        superlatives = ['BEST', 'AMAZING', 'INCREDIBLE', 'PERFECT', 'EXCELLENT', 'FANTASTIC', 'WONDERFUL', 'OUTSTANDING', 'AWESOME', 'GREAT', 'WORST', 'TERRIBLE', 'HORRIBLE', 'DISGUSTING']
        found_superlatives = [word for word in superlatives if word in text_upper]
        if len(found_superlatives) >= 2:  # Lowered threshold
            template_indicators.append(f"Multiple superlatives: {', '.join(found_superlatives)}")
        
        # Short length with excessive praise/criticism
        word_count = len(review_text.split())
        if word_count <= 12 and (len(found_superlatives) >= 2 or exclamation_count >= 3):
            template_indicators.append("Very short but overly emotional language")
        
        # Repetitive words pattern
        word_freq = {}
        for word in words:
            clean_word = word.strip('!.,?').upper()
            if len(clean_word) > 3:
                word_freq[clean_word] = word_freq.get(clean_word, 0) + 1
        
        repeated_words = [word for word, count in word_freq.items() if count >= 2]
        if repeated_words:
            template_indicators.append(f"Repetitive words: {', '.join(repeated_words[:3])}")
        
        return template_indicators

    def _create_training_enhanced_prompt(self, review_text, rating, user_metadata, business_name):
        """Create enhanced prompt with comprehensive training data insights"""
        
        # Check for template patterns
        template_patterns = self._detect_template_patterns(review_text)
        template_context = ""
        if template_patterns:
            template_context = f"""
⚠️ TEMPLATE LANGUAGE DETECTED:
{chr(10).join(f"- {pattern}" for pattern in template_patterns)}
"""
        
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
        
        prompt = f"""You are an expert review authenticity classifier. Analyze this review and classify it into one of the detailed subcategories below.

� DETAILED FAKE REVIEW CLASSIFICATION SYSTEM:

=== LEGITIMATE REVIEWS ===
LEGITIMATE: Genuine reviews with personal experiences, specific details, balanced opinions, natural language

=== MARKETING/COMMERCIAL FAKE REVIEWS ===
ADVERTISEMENT: Direct ads disguised as reviews, promoting services/products
PROMOTIONAL_CONTENT: Marketing copy, sales pitches, promotional language
BUSINESS_OWNER_FAKE: Owner pretending to be customer, overly defensive responses
COMPETITOR_ATTACK: Malicious reviews targeting competitors with false claims

=== CONTENT QUALITY ISSUES ===
TOO_GENERIC: Vague, could apply to any business ("great place", "good service")
TOO_LITTLE_DETAIL: Extremely short, no specific information or context
TEMPLATE_LANGUAGE: Copy-paste template patterns, excessive exclamation marks (!!!), ALL CAPS words, generic superlatives (BEST, AMAZING, INCREDIBLE, PERFECT), repetitive praise words, no specific details
UNREALISTIC_PRAISE: Impossibly perfect experiences, over-the-top language
COPY_PASTE_REVIEW: Identical or near-identical to other reviews

=== EXPERIENCE ISSUES ===
NO_ACTUAL_EXPERIENCE: User admits they haven't visited ("haven't been there")
HEARSAY_REVIEW: Based on others' experiences ("my friend said", "I heard")
PLANNING_TO_VISIT: Future plans, not actual experiences ("planning to go")
LOCATION_IMPOSSIBLE: Review doesn't match business location/services

=== RATING MANIPULATION ===
RATING_TEXT_MISMATCH: Star rating contradicts review text sentiment
EXTREME_RATING_ABUSE: 1 or 5 stars with mild/neutral content
SENTIMENT_CONTRADICTION: Positive words with negative rating or vice versa

=== SPAM PATTERNS ===
REPETITIVE_SPAM: Same user posting identical/similar reviews
KEYWORD_STUFFING: Unnatural keyword repetition for SEO
CONTACT_INFO_SPAM: Phone numbers, emails, websites, "call now"
LINK_SPAM: URLs, social media handles, promotional links

=== BEHAVIORAL ISSUES ===
BOT_GENERATED: AI-generated text, unnatural language patterns
MASS_GENERATED: Part of coordinated fake review campaign
INCENTIVIZED_FAKE: Reviews for incentives, rewards, payments
PAID_REVIEW_SERVICE: Professional fake review services

=== CONTENT PROBLEMS ===
UNHELPFUL_CONTENT: No useful information for potential customers
IRRELEVANT_CONTENT: Off-topic, unrelated to business/service
NONSENSICAL_TEXT: Gibberish, random characters, broken language
WRONG_BUSINESS: Review about different business/location

=== SUSPICIOUS (FLAG FOR REVIEW) ===
SUSPICIOUS_USER_PATTERN: Unusual posting patterns, timing anomalies
IMAGE_TEXT_MISMATCH: Photos don't match review description
TEMPORAL_ANOMALY: Review timing doesn't match business operations
UNUSUAL_LANGUAGE_PATTERN: Odd grammar, non-native inconsistencies
REVIEW TO ANALYZE:
{template_context}
{user_context}
{business_context}
{rating_context}

Review Text: "{review_text}"

Analyze this review carefully and respond EXACTLY in this format:
CATEGORY: [Choose the MOST SPECIFIC category from the list above that best describes this review]
CONFIDENCE: [0.000-1.000]
REASONING: [Brief explanation of why you classified it this way, mention specific indicators]

⚠️ CLASSIFICATION GUIDELINES:
- LEGITIMATE: Only if review shows genuine personal experience with specific details
- TEMPLATE_LANGUAGE: Multiple ALL CAPS words, excessive !!!, generic superlatives without specifics
- CONTACT_INFO_SPAM: Phone numbers, websites, "contact us", promotional language
- NO_ACTUAL_EXPERIENCE: "haven't been", "heard from friend", admitting no visit
- RATING_TEXT_MISMATCH: 5-star with negative content OR 1-star with positive content
- TOO_GENERIC: "Great place", "good service" without any specifics
- If template language patterns are detected above, strongly consider TEMPLATE_LANGUAGE category

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
