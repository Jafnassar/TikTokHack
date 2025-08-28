import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import json
import re
from typing import List, Dict, Any
import os

#MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct" 

#using tokenizer to convert raw text to works
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.float16,
)

#add on to the policy y'alls :D
# Prompt engineers please changeee :)
def create_prompt(review_text: str) -> str:
    return f"""
You have to check for restaurant reviews. 
Given the following review, classify whether it violates each of these policies:

1. No advertisements (links, promotions).
2. No irrelevant content which are unrelated to the restaurant.
3. No rants from users who have not visited.

Respond ONLY in JSON with this schema:
{{
  "violation_found": <true/false>,
  "violation_types": {{
    "advertisement": {{"violated": <true/false>, "confidence": <0-1>, "explanation": "<string>"}},
    "irrelevant": {{"violated": <true/false>, "confidence": <0-1>, "explanation": "<string>"}},
    "rant": {{"violated": <true/false>, "confidence": <0-1>, "explanation": "<string>"}}
  }},
  "overall_quality_score": <0-1>,
  "final_explanation": "<string>"
}}

[Review STARTS]
{review_text}
[Review ENDS]
"""

def extract_json_from_response(text: str) -> Dict[str, Any]:
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            return {
                "violation_found": True,
                "violation_types": {
                    "advertisement": {"violated": False, "confidence": 0, "explanation": "Failed to parse model response"},
                    "irrelevant": {"violated": False, "confidence": 0, "explanation": "Failed to parse model response"},
                    "rant": {"violated": False, "confidence": 0, "explanation": "Failed to parse model response"}
                },
                "overall_quality_score": 0,
                "final_explanation": "Failed to parse model response"
            }

    # Default if no JSON found
    return {
        "violation_found": True,
        "violation_types": {
            "advertisement": {"violated": False, "confidence": 0, "explanation": "No valid response from model"},
            "irrelevant": {"violated": False, "confidence": 0, "explanation": "No valid response from model"},
            "rant": {"violated": False, "confidence": 0, "explanation": "No valid response from model"}
        },
        "overall_quality_score": 0,
        "final_explanation": "No valid response from model"
    }

def analyze_review(review_text: str) -> Dict[str, Any]:
    prompt = create_prompt(review_text)
    response = pipe(
        prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        return_full_text=False
    )
    
    generated_text = response[0]['generated_text']
    result = extract_json_from_response(generated_text)
    print(result)
    result['review_text'] = review_text
    return result

def analyze_reviews_batch(reviews: List[str], batch_size: int = 5) -> List[Dict[str, Any]]:
    results = []
    for i in range(0, len(reviews), batch_size):
        batch = reviews[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(reviews)-1)//batch_size + 1}")
        
        for review in batch:
            try:
                result = analyze_review(review)
                results.append(result)
            except Exception as e:
                print(f"Error processing review: {e}")
                results.append({
                    "review_text": review,
                    "violation_found": True,
                    "error": str(e)
                })
    return results

def generate_report(results: List[Dict[str, Any]]) -> pd.DataFrame:
    report_data = []
    
    for result in results:
        if 'error' in result:
            continue
            
        row = {
            'review_text': result['review_text'],
            'violation_found': result['violation_found'],
            'overall_quality_score': result['overall_quality_score'],
            'final_explanation': result['final_explanation']
        }
        
        for violation_type, details in result['violation_types'].items():
            row[f'{violation_type}_violated'] = details['violated']
            row[f'{violation_type}_confidence'] = details['confidence']
            row[f'{violation_type}_explanation'] = details['explanation']
        
        report_data.append(row)
    
    return pd.DataFrame(report_data)

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("./data/reviews_cleaned.csv")
    
    # Use only a subset for testing
    all_reviews = df["text"].dropna().tolist()

    results = analyze_reviews_batch(all_reviews, batch_size=2)
    report = generate_report(results)

    print("\nAnalysis Report (summary):")
    print(report[['review_text', 'violation_found', 'overall_quality_score']])

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "review_analysis_report.csv")
    report.to_csv(output_file, index=False)
    print(f"\nFull report saved to '{output_file}'")

    print("\nDetailed analysis for the first review:")
    print(json.dumps(results[0], indent=2))

