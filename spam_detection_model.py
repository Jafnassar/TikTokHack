"""
Spam Review Detection Model
This module creates and trains a machine learning model to detect spam/fake reviews
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
import pickle
import re
import nltk
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class SpamReviewDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.feature_extractors = {}
        
    def extract_features(self, text):
        """Extract hand-crafted features from review text"""
        features = {}
        
        # Basic text statistics
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        features['sentence_count'] = len(text.split('.'))
        
        # Punctuation and capitalization
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['punctuation_ratio'] = sum(1 for c in text if c in '!@#$%^&*()_+-=[]{}|;:,.<>?') / len(text) if text else 0
        
        # Repetitive patterns
        features['repeated_chars'] = len(re.findall(r'(.)\1{2,}', text.lower()))
        features['repeated_words'] = len(re.findall(r'\b(\w+)\s+\1\b', text.lower()))
        
        # Enhanced fake review detection patterns
        features.update(self._detect_advanced_spam_patterns(text))
        
        # Promotional language indicators
        promo_words = ['amazing', 'incredible', 'unbelievable', 'fantastic', 'awesome', 'perfect', 
                      'best', 'worst', 'terrible', 'horrible', 'outstanding', 'excellent',
                      'cheap', 'expensive', 'free', 'discount', 'deal', 'offer', 'sale']
        features['promo_word_count'] = sum(1 for word in promo_words if word in text.lower())
        
        # Contact information patterns
        features['has_phone'] = len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text))
        features['has_email'] = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        features['has_website'] = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
        
        # Sentiment analysis
        try:
            blob = TextBlob(text)
            features['sentiment_polarity'] = blob.sentiment.polarity
            features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        except:
            features['sentiment_polarity'] = 0
            features['sentiment_subjectivity'] = 0
        
        # Urgency indicators
        urgency_words = ['urgent', 'hurry', 'limited', 'now', 'today', 'immediately', 'quickly', 'fast']
        features['urgency_count'] = sum(1 for word in urgency_words if word in text.lower())
        
        # Personal endorsement indicators
        personal_words = ['recommend', 'suggest', 'tell', 'friends', 'family', 'everyone', 'mention']
        features['personal_endorsement'] = sum(1 for word in personal_words if word in text.lower())
        
        return features
    
    def _detect_advanced_spam_patterns(self, text):
        """Detect advanced spam patterns (moved from dashboard)"""
        features = {}
        text_lower = text.lower()
        
        # Contact patterns
        contact_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
            r'\bcall\s+\d+\b',                 # "call 555..."
            r'\bbook\s+your\s+\w+\b',          # "book your consultation"
            r'\bschedule\s+\w+\b',             # "schedule appointment"
        ]
        features['contact_pattern_count'] = sum(1 for pattern in contact_patterns if re.search(pattern, text_lower))
        
        # Promotional language patterns
        promotional_patterns = [
            r'\blimited[- ]?time\b',           # "limited-time"
            r'\boffer\s+ends\b',               # "offer ends"
            r'\bfree\s+\w+\s+with\s+purchase\b', # "free X with purchase"
            r'\bpackage\s+deal\b',             # "package deal"
            r'\bthank\s+me\s+later\b',         # "thank me later"
            r'\bdo\s+yourself\s+a\s+favor\b',  # "do yourself a favor"
            r'\bunmatched\s+quality\b',        # "unmatched quality"
            r'\bmention\s+this\s+review\b',    # "mention this review"
            r'\bif\s+you\s+mention\b',         # "if you mention"
            r'\ball\s+month\b',                # "all month"
            r'\bspecial\s+offer\b',            # "special offer"
            r'\bonly\s+\$?\d+\.?\d*\b',        # "only $29.95"
            r'\bjust\s+\$?\d+\.?\d*\b',        # "just $29.95"
        ]
        features['promotional_pattern_count'] = sum(1 for pattern in promotional_patterns if re.search(pattern, text_lower))
        
        # Suspicious endorsement patterns
        endorsement_patterns = [
            r'\bshoutout\s+to\s+\w+\b',        # "shoutout to Mark"
            r'\bmanager.*went\s+above\b',      # "manager went above and beyond"
            r'\bwent\s+above\s+and\s+beyond\b', # "went above and beyond"
            r'\bhad\s+no\s+idea\s+they\b',     # "had no idea they offered"
            r'\bonly\s+place\s+I.?m?\s+trusting\b', # "only place I'm trusting"
            r'\bthis\s+is\s+the\s+only\s+place\b', # "this is the only place"
            r'\bnever\s+going\s+anywhere\s+else\b', # "never going anywhere else"
            r'\b\w+\s+in\s+\w+\s+found\b',     # "Mike in diagnostics found"
        ]
        features['endorsement_pattern_count'] = sum(1 for pattern in endorsement_patterns if re.search(pattern, text_lower))
        
        # Urgency tactics patterns
        urgency_patterns = [
            r'\bbefore.*ends\s+\w+\b',         # "before offer ends Friday"
            r'\bends\s+\w+day\b',              # "ends Friday"
            r'\bhurry\b',                      # "hurry"
            r'\bwhile\s+supplies\s+last\b',    # "while supplies last"
            r'\bthis\s+month\s+only\b',        # "this month only"
            r'\blimited\s+time\b',             # "limited time"
        ]
        features['urgency_pattern_count'] = sum(1 for pattern in urgency_patterns if re.search(pattern, text_lower))
        
        # Pricing mention patterns
        pricing_patterns = [
            r'\$\d+\.?\d*',                    # Any price mention like $29.95
            r'\bfree\s+\w+\b',                 # "free diagnostic"
            r'\bonly\s+charged\s+\w+\s+for\b', # "only charged me for"
            r'\brunning\s+for\s+just\b',       # "running for just"
            r'\bcost\s+me\s+nothing\b',        # "cost me nothing"
        ]
        features['pricing_pattern_count'] = sum(1 for pattern in pricing_patterns if re.search(pattern, text_lower))
        
        # Business-specific patterns
        business_patterns = [
            r'\b\w+\s+at\s+[A-Z][a-z]+\s+[A-Z&][a-z]*\b', # "Mike at Precision Auto"
            r'\bguys\s+at\s+[A-Z]\w+\b',       # "guys at Precision"
            r'\bteam\s+at\s+[A-Z]\w+\b',       # "team at Business"
            r'\bstaff\s+at\s+[A-Z]\w+\b',      # "staff at Business"
        ]
        features['business_pattern_count'] = sum(1 for pattern in business_patterns if re.search(pattern, text))
        
        # Excessive superlatives detection (enhanced)
        superlatives = ['incredible', 'amazing', 'unmatched', 'insane', 'top-notch', 'unbelievable', 
                       'blown away', 'awesome', 'fantastic', 'outstanding', 'perfect', 'excellent', 
                       'spectacular', 'phenomenal', 'extraordinary', 'magnificent', 'superb']
        features['superlative_count'] = sum(1 for word in superlatives if word in text_lower)
        features['has_excessive_superlatives'] = 1 if features['superlative_count'] >= 3 else 0
        
        # Calculate total spam signal score
        features['total_spam_signals'] = (
            features['contact_pattern_count'] +
            features['promotional_pattern_count'] +
            features['endorsement_pattern_count'] +
            features['urgency_pattern_count'] +
            features['pricing_pattern_count'] +
            features['business_pattern_count'] +
            features['has_excessive_superlatives']
        )
        
        return features
    
    def generate_synthetic_spam_reviews(self, legitimate_reviews, num_spam=500):
        """Generate synthetic spam reviews for training"""
        
        # More realistic spam templates based on actual patterns
        spam_templates = [
            # Promotional spam with contact info
            "Absolutely incredible! Shoutout to {name}, the manager, who went above and beyond to make sure I got the limited-time \"{product}\" package deal. I had no idea they offered free {service} with purchase! The quality is unmatched and the value is insane. Do yourself a favor and call {phone} to book your consultation before the offer ends {day}. You can thank me later!",
            
            "My {item} started making that awful noise again and I dreaded the repair bill. Took it to the guys at {business} and was blown away. Instead of pushing for a full {expensive_service}, {tech_name} in diagnostics found it was just a {simple_fix}. They fixed it in under an hour and only charged me for the diagnostic, which they're running for just ${price} all month if you mention this review. This is the only place I'm trusting with my {item} from now on!",
            
            "Stumbled in here to escape the rain and found my new happy place. The atmosphere is so cozy, and the staff is the friendliest. I tried the new {product}—it's literally life-changing. And you HAVE to get the {food_item}, it's honestly the best in the city. The best part? They just launched their rewards app. I downloaded it right at the table and got my second {item} completely free. You just scan the QR code on the menu. Seriously, why would you go anywhere else?",
            
            # Business endorsement spam
            "Amazing service! Call {phone} for special discount! Best deals in town, hurry up! Limited time only!",
            "Outstanding quality! Email us at {email} for exclusive deals. Perfect place! Mention this review for {percent}% off!",
            "Fantastic experience! Visit our website {website} for more offers. Thank me later!",
            
            # Fake positive reviews
            "This place is absolutely amazing!!! Best food ever, outstanding service, incredible atmosphere! Perfect experience!",
            "INCREDIBLE RESTAURANT!!! BEST QUALITY EVER!!! AMAZING!!! PERFECT!!! OUTSTANDING!!! FANTASTIC!!! UNBELIEVABLE!!!",
            "Best place in town! Amazing quality, fantastic service, incredible value! Highly recommend! Outstanding!",
            "Perfect restaurant! Best service ever, amazing food, incredible atmosphere! Must visit! Fantastic experience!",
            
            # Competitor attack spam  
            "Terrible experience! Worst food ever! Horrible service! Don't waste your money here! Awful place!",
            "Bad restaurant! Awful food, terrible staff, horrible value! Stay away! Worst experience ever!",
            
            # Subtle promotional spam
            "Great experience! The staff really knows their stuff. {name} helped me find exactly what I needed and even told me about their current promotion. Definitely coming back!",
            "Really satisfied with the service here. They have this new {service} that's only available this month. Worth checking out if you're in the area.",
            
            # Review incentive spam
            "Love this place! By the way, if you write a review here, you get {discount}% off your next visit. Just thought I'd mention it since it's a win-win!",
            "Five stars! Pro tip: mention you saw this review when you visit and they'll hook you up with their special {offer}. Trust me on this one!",
        ]
        
        spam_reviews = []
        names = ["Mike", "Sarah", "John", "Lisa", "Dave", "Amy", "Chris", "Jessica", "Mark", "Rachel"]
        businesses = ["Precision Auto", "Elite Services", "Premium Solutions", "Quality Plus", "Expert Care"]
        products = ["Summer Special", "Premium Package", "Elite Bundle", "Professional Service", "Deluxe Treatment"]
        services = ["installation", "consultation", "diagnostic", "maintenance", "setup"]
        days = ["Friday", "Monday", "this weekend", "next week"]
        items = ["car", "computer", "appliance", "system", "equipment"]
        
        for i in range(num_spam):
            template = np.random.choice(spam_templates)
            
            # Fill in template variables
            filled_template = template.format(
                name=np.random.choice(names),
                business=np.random.choice(businesses),
                product=np.random.choice(products),
                service=np.random.choice(services),
                day=np.random.choice(days),
                item=np.random.choice(items),
                tech_name=np.random.choice(names),
                expensive_service=np.random.choice(["engine replacement", "full overhaul", "complete rebuild"]),
                simple_fix=np.random.choice(["loose connection", "minor adjustment", "quick calibration"]),
                price=np.random.choice(["29.95", "19.99", "39.95", "49.95"]),
                food_item=np.random.choice(["avocado toast", "specialty coffee", "signature dish"]),
                phone=f"{np.random.randint(100,999)}-{np.random.randint(1000,9999)}",
                email=f"contact{i}@business{np.random.randint(1,100)}.com",
                website=f"www.business{np.random.randint(1,100)}.com",
                percent=np.random.choice([10, 15, 20, 25, 30, 50]),
                discount=np.random.choice([10, 15, 20]),
                offer=np.random.choice(["discount", "bonus service", "free upgrade"])
            )
            
            # Add some variations to make it more realistic
            variations = [
                lambda x: x + " " + np.random.choice(["Amazing!", "Perfect!", "Highly recommend!", "Must try!", "Best ever!"]),
                lambda x: x.replace("!", "!!!") if np.random.random() < 0.3 else x,
                lambda x: x.upper() if np.random.random() < 0.05 else x,  # Occasional all caps
                lambda x: x + " " + np.random.choice(["Thanks again!", "Five stars!", "Will be back!", "Love this place!"]),
            ]
            
            if np.random.random() < 0.7:  # 70% chance to apply variation
                variation = np.random.choice(variations)
                filled_template = variation(filled_template)
            
            spam_reviews.append({
                'text': filled_template,
                'rating': np.random.choice([1, 1, 5, 5, 5, 5]),  # Heavily biased towards extreme ratings
                'author_name': f"User{i}",
                'business_name': f"Business{np.random.randint(1,20)}",
                'is_spam': 1
            })
        
        return spam_reviews
    
    def prepare_training_data(self, reviews_df):
        """Prepare training data with both legitimate and spam reviews"""
        # Mark all current reviews as legitimate
        legitimate_data = []
        for _, row in reviews_df.iterrows():
            legitimate_data.append({
                'text': row['text'],
                'rating': row['rating'],
                'author_name': row['author_name'],
                'business_name': row['business_name'],
                'is_spam': 0
            })
        
        # Generate synthetic spam reviews
        spam_data = self.generate_synthetic_spam_reviews(legitimate_data, num_spam=len(legitimate_data))
        
        # Combine legitimate and spam data
        all_data = legitimate_data + spam_data
        
        # Create DataFrame
        training_df = pd.DataFrame(all_data)
        
        print(f"Training data prepared:")
        print(f"Legitimate reviews: {len(legitimate_data)}")
        print(f"Spam reviews: {len(spam_data)}")
        print(f"Total reviews: {len(all_data)}")
        print(f"Spam ratio: {len(spam_data)/len(all_data):.2%}")
        
        return training_df
    
    def train_model(self, legitimate_reviews_df, num_spam_samples=500):
        """Train the spam detection model"""
        print("Training spam detection model...")
        
        # Generate synthetic spam reviews
        print(f"Generating {num_spam_samples} synthetic spam reviews...")
        spam_reviews = self.generate_synthetic_spam_reviews(legitimate_reviews_df, num_spam_samples)
        
        # Combine legitimate and spam reviews
        legitimate_data = []
        for _, row in legitimate_reviews_df.iterrows():
            legitimate_data.append({
                'text': row['text'],
                'rating': row['rating'],
                'is_spam': 0
            })
        
        all_reviews = legitimate_data + spam_reviews
        training_df = pd.DataFrame(all_reviews)
        
        print(f"Training on {len(legitimate_data)} legitimate + {len(spam_reviews)} spam reviews")
        
        # Extract features for each review
        print("Extracting features...")
        feature_data = []
        for _, row in training_df.iterrows():
            features = self.extract_features(row['text'])
            features['rating'] = row['rating']
            features['text'] = row['text']
            features['is_spam'] = row['is_spam']
            feature_data.append(features)
        
        feature_df = pd.DataFrame(feature_data)
        
        # Prepare features
        text_features = feature_df['text'].values
        numeric_features = feature_df.drop(['text', 'is_spam'], axis=1).values
        labels = feature_df['is_spam'].values
        
        # Create TF-IDF features
        print("Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            min_df=2,
            max_df=0.95
        )
        
        tfidf_features = self.vectorizer.fit_transform(text_features).toarray()
        
        # Combine TF-IDF and numeric features
        combined_features = np.hstack([tfidf_features, numeric_features])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            combined_features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train multiple models and select the best one
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        best_score = 0
        best_model_name = ""
        
        print("Training and evaluating models...")
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            avg_score = cv_scores.mean()
            
            print(f"{name}: CV AUC = {avg_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            if avg_score > best_score:
                best_score = avg_score
                best_model_name = name
        
        # Train the best model
        print(f"\nBest model: {best_model_name}")
        self.model = models[best_model_name]
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        print(f"\nTest Set Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Spam']))
        
        # Store feature names for later use
        self.feature_names = (
            list(self.vectorizer.get_feature_names_out()) + 
            list(feature_df.drop(['text', 'is_spam'], axis=1).columns)
        )
        
        return self.model
    
    def predict_spam(self, text, rating=None):
        """Predict if a review is spam with enhanced signal-based logic"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Extract features
        features = self.extract_features(text)
        if rating is not None:
            features['rating'] = rating
        else:
            features['rating'] = 3  # Default neutral rating
        
        # Create TF-IDF features
        tfidf_features = self.vectorizer.transform([text]).toarray()
        
        # Prepare numeric features in the same order as training
        numeric_feature_names = [col for col in self.feature_names if col not in self.vectorizer.get_feature_names_out()]
        numeric_features = np.array([[features.get(name, 0) for name in numeric_feature_names]])
        
        # Combine features
        combined_features = np.hstack([tfidf_features, numeric_features])
        
        # Get original model prediction
        original_spam_probability = self.model.predict_proba(combined_features)[0][1]
        
        # Count spam signals for enhanced logic
        spam_signals = self._count_spam_signals_from_features(features)
        
        # Enhanced decision logic - adjust probability based on signal count
        adjusted_probability = original_spam_probability
        
        if spam_signals >= 10:
            # If we detect 10+ spam signals, heavily weight towards spam
            adjusted_probability = max(original_spam_probability, 0.85)
        elif spam_signals >= 7:
            # 7-9 signals should be at least 70% spam probability
            adjusted_probability = max(original_spam_probability, 0.70)
        elif spam_signals >= 5:
            # 5-6 signals should be at least 60% spam probability
            adjusted_probability = max(original_spam_probability, 0.60)
        elif spam_signals >= 3:
            # 3-4 signals should be at least 50% spam probability
            adjusted_probability = max(original_spam_probability, 0.50)
        
        # Determine if spam with signal-based thresholds
        if spam_signals >= 8:
            is_spam = adjusted_probability > 0.6  # Lower threshold for high signal counts
        elif spam_signals >= 5:
            is_spam = adjusted_probability > 0.65  # Moderate threshold
        else:
            is_spam = adjusted_probability > 0.7   # Higher threshold for low signals
        
        # Get confidence
        confidence = max(adjusted_probability, 1 - adjusted_probability)
        
        # Generate explanation based on features
        explanation_parts = []
        
        if features['promo_word_count'] > 2:
            explanation_parts.append(f"High promotional language ({features['promo_word_count']} promotional words)")
        
        if features['contact_pattern_count'] > 0:
            explanation_parts.append("Contains contact information patterns")
        
        if features['promotional_pattern_count'] > 0:
            explanation_parts.append(f"Contains promotional language patterns ({features['promotional_pattern_count']} detected)")
        
        if features['endorsement_pattern_count'] > 0:
            explanation_parts.append(f"Contains suspicious endorsement patterns ({features['endorsement_pattern_count']} detected)")
        
        if features['urgency_pattern_count'] > 0:
            explanation_parts.append(f"Contains urgency tactics ({features['urgency_pattern_count']} detected)")
        
        if features['pricing_pattern_count'] > 0:
            explanation_parts.append(f"Contains pricing mentions ({features['pricing_pattern_count']} detected)")
        
        if features['business_pattern_count'] > 0:
            explanation_parts.append(f"Contains business-specific patterns ({features['business_pattern_count']} detected)")
        
        if features['has_phone'] > 0 or features['has_email'] > 0 or features['has_website'] > 0:
            explanation_parts.append("Contains contact information")
        
        if features['caps_ratio'] > 0.3:
            explanation_parts.append("Excessive capitalization")
        
        if features['exclamation_count'] > 3:
            explanation_parts.append("Excessive exclamation marks")
        
        if features['urgency_count'] > 1:
            explanation_parts.append("Urgency language detected")
        
        if features['personal_endorsement'] > 2:
            explanation_parts.append("Suspicious endorsement language")
        
        if features['has_excessive_superlatives']:
            explanation_parts.append(f"Excessive superlative language ({features['superlative_count']} superlatives)")
        
        if features['total_spam_signals'] > 2:
            explanation_parts.append(f"Multiple spam signals detected (total score: {features['total_spam_signals']})")
        
        # Add signal count to explanation
        explanation_parts.append(f"Total spam signals: {spam_signals}")
        
        if len(explanation_parts) == 0:
            if is_spam:
                explanation_parts.append("Pattern matches known spam characteristics")
            else:
                explanation_parts.append("Review appears legitimate")
        
        explanation = "; ".join(explanation_parts)
        
        return {
            'is_spam': bool(is_spam),
            'spam_probability': float(adjusted_probability),
            'original_probability': float(original_spam_probability),
            'confidence': float(confidence),
            'explanation': explanation,
            'spam_signals_count': spam_signals,
            'features_detected': features
        }
    
    def _count_spam_signals_from_features(self, features):
        """Count spam signals from extracted features"""
        signals = 0
        
        # Contact information (strong signals)
        if features.get('has_phone', 0) > 0:
            signals += 2
        if features.get('has_email', 0) > 0:
            signals += 2
        if features.get('has_website', 0) > 0:
            signals += 2
            
        # Pattern counts
        signals += features.get('contact_pattern_count', 0)
        signals += features.get('promotional_pattern_count', 0)
        signals += features.get('endorsement_pattern_count', 0)
        signals += features.get('urgency_pattern_count', 0)
        signals += features.get('pricing_pattern_count', 0)
        signals += features.get('business_pattern_count', 0)
        
        # Language indicators
        if features.get('promo_word_count', 0) > 2:
            signals += 1
        if features.get('caps_ratio', 0) > 0.3:
            signals += 1
        if features.get('exclamation_count', 0) > 3:
            signals += 1
        if features.get('urgency_count', 0) > 1:
            signals += 1
        if features.get('personal_endorsement', 0) > 2:
            signals += 1
        if features.get('has_excessive_superlatives', False):
            signals += 1
            
        # Extreme ratings with generic content
        if features.get('rating', 3) in [1, 5] and features.get('superlative_count', 0) >= 3:
            signals += 1
            
        return signals
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.feature_names = model_data['feature_names']
        
        print(f"Model loaded from {filepath}")

def main():
    """Train and save the spam detection model"""
    # Load the reviews dataset
    print("Loading reviews dataset...")
    reviews_df = pd.read_csv('reviews_cleaned.csv')
    
    # Initialize detector
    detector = SpamReviewDetector()
    
    # Prepare training data
    training_df = detector.prepare_training_data(reviews_df)
    
    # Train model
    detector.train_model(training_df)
    
    # Save model
    detector.save_model('spam_detection_model.pkl')
    
    # Test with some examples
    print("\n" + "="*50)
    print("TESTING THE MODEL")
    print("="*50)
    
    test_reviews = [
        "Great food and excellent service! Really enjoyed my meal here.",
        "Amazing place!!! Call 555-123-4567 for special discount! Best deals in town, hurry up!!!",
        "The pizza was okay, nothing special. Service was a bit slow but friendly.",
        "INCREDIBLE RESTAURANT!!! BEST FOOD EVER!!! AMAZING!!! PERFECT!!! OUTSTANDING!!!",
        "I had dinner here last week. The atmosphere was nice and the staff was helpful."
    ]
    
    for i, review in enumerate(test_reviews, 1):
        result = detector.predict_spam(review)
        print(f"\nTest {i}: {review[:50]}...")
        print(f"Spam: {result['is_spam']} (probability: {result['spam_probability']:.3f})")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Explanation: {result['explanation']}")

if __name__ == "__main__":
    main()
