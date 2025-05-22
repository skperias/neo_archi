# test_claude35.py
# Diagnostic tool to test Claude 3.5 Sonnet model

import boto3
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

def test_claude35_sonnet(region="ap-south-1"):
    """Test connection to the Claude 3.5 Sonnet model specifically."""
    model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    print(f"\n--- Testing Claude 3.5 Sonnet Model Specifically ---\n")
    
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

    # Correct format for Claude 3.5
    message = "Create a brief introduction for a system architecture document."
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.999,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": message
                    }
                ]
            }
        ]
    })
    
    print("Using Claude 3.5 Sonnet format:")
    print(f"Request payload: {body}\n")
    
    # Invoke the model
    try:
        start_time = time.time()
        response = client.invoke_model(body=body, modelId=model_id)
        end_time = time.time()
        
        print(f"✓ Model invocation successful! (Took {end_time - start_time:.2f} seconds)")
        
        # Parse the response
        response_body = json.loads(response.get('body').read())
        print(f"\nRaw response: {json.dumps(response_body, indent=2)}")
        
        # Extract text from the response
        result = ""
        if 'content' in response_body and len(response_body['content']) > 0:
            for content_item in response_body['content']:
                if content_item.get('type') == 'text':
                    result += content_item.get('text', '')
        
        if result:
            print(f"\n--- Generated Response ---\n{result}")
            print("\n--- Test Successful! ---")
            return True
        else:
            print("\n✗ Could not extract text from response")
            print("\n--- Test Failed ---")
            return False
            
    except Exception as e:
        print(f"\n✗ Failed to invoke model: {e}")
        print("\n--- Test Failed ---")
        return False

if __name__ == "__main__":
    test_claude35_sonnet()
