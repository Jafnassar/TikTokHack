#!/usr/bin/env python3
"""
Test simple Qwen prompt
"""

import sys
sys.path.append('.')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print('🔍 Testing Simple Qwen Prompt')

try:
    # Load model
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Simple test prompt
    prompt = """Classify this review as LEGITIMATE or SPAM:
Review: "Text (555) 123-PAWS to book your appointment now!"

Classification:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response[len(prompt):].strip()
    
    print(f'Prompt: {prompt}')
    print(f'Response: "{response}"')
    
    if 'SPAM' in response:
        print('✅ Model correctly detected advertisement!')
    else:
        print('❌ Model did not detect advertisement')

except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
