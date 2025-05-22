# test_bedrock.py
# Diagnostic tool to test Bedrock model connections

import argparse
import boto3
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

def test_bedrock_model(model_id, region, message="Hello, can you generate a short test response?"):
    """Test connection to a Bedrock model and verify response."""
    print(f"\n--- Testing AWS Bedrock Model: {model_id} ---\n")
    
    # Check environment variables
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID") 
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not aws_access_key or not aws_secret_key:
        print("Warning: AWS credentials not found in environment variables.")
        print("Will attempt to use default credentials from AWS configuration.")

    # Create Bedrock client
    try:
        client = boto3.client(
            'bedrock-runtime',
            region_name=region
        )
        print(f"✓ Successfully created Bedrock client for region {region}")
    except Exception as e:
        print(f"✗ Failed to create Bedrock client: {e}")
        return False

    # Determine the correct payload format based on model
    if "anthropic" in model_id.lower():
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": message}]
        })
        print("✓ Using Anthropic Claude format")
    elif "meta" in model_id.lower() or "llama" in model_id.lower():
        body = json.dumps({
            "prompt": message,
            "max_gen_len": 100,
            "temperature": 0.7
        })
        print("✓ Using Meta Llama format")
    elif "titan" in model_id.lower() or "amazon" in model_id.lower():
        body = json.dumps({
            "inputText": message,
            "textGenerationConfig": {
                "maxTokenCount": 100,
                "temperature": 0.7,
                "topP": 0.9
            }
        })
        print("✓ Using Amazon Titan format")
    elif "cohere" in model_id.lower():
        body = json.dumps({
            "prompt": message,
            "max_tokens": 100,
            "temperature": 0.7
        })
        print("✓ Using Cohere format")
    else:
        print(f"! Unknown model type: {model_id}")
        print("  Using generic format - this may fail")
        body = json.dumps({
            "prompt": message,
            "max_tokens": 100
        })

    # Invoke the model
    try:
        print(f"\nAttempting to invoke model '{model_id}'...")
        print(f"Request payload: {body}\n")
        
        start_time = time.time()
        response = client.invoke_model(body=body, modelId=model_id)
        end_time = time.time()
        
        print(f"✓ Model invocation successful! (Took {end_time - start_time:.2f} seconds)")
        
        # Parse the response
        response_body = json.loads(response.get('body').read())
        print(f"\nRaw response: {json.dumps(response_body, indent=2)}")
        
        # Extract the generated text based on model type
        if "anthropic" in model_id.lower():
            if 'content' in response_body and len(response_body['content']) > 0:
                result = response_body['content'][0]['text']
            else:
                result = "Could not extract text from Anthropic response"
        elif "meta" in model_id.lower() or "llama" in model_id.lower():
            result = response_body.get('generation', "Could not extract text from Meta response")
        elif "titan" in model_id.lower() or "amazon" in model_id.lower():
            if 'results' in response_body and len(response_body['results']) > 0:
                result = response_body['results'][0]['outputText']
            else:
                result = "Could not extract text from Amazon response"
        elif "cohere" in model_id.lower():
            if 'generations' in response_body and len(response_body['generations']) > 0:
                result = response_body['generations'][0]['text']
            else:
                result = "Could not extract text from Cohere response"
        else:
            # Try to find text in the response
            for key in ['text', 'content', 'generated_text', 'response', 'output', 'completion']:
                if key in response_body:
                    result = response_body[key]
                    break
            else:
                result = "Could not extract text from response"
                
        print(f"\n--- Generated Response ---\n{result}")
        print("\n--- Test Successful! ---")
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to invoke model: {e}")
        print("\n--- Test Failed ---")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test AWS Bedrock model connections')
    parser.add_argument('--model', type=str, default="anthropic.claude-3-sonnet-20240229-v1:0", 
                        help='Bedrock model ID to test')
    parser.add_argument('--region', type=str, default="us-east-1",
                        help='AWS region where Bedrock is available')
    parser.add_argument('--message', type=str, 
                        default="Generate a very short hello world message.",
                        help='Test message to send to the model')
    
    args = parser.parse_args()
    test_bedrock_model(args.model, args.region, args.message)

if __name__ == "__main__":
    main()
