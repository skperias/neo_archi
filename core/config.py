# /core/config.py
import os
from dotenv import load_dotenv

load_dotenv()

def get_config():
    return {
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "aws_region": os.getenv("AWS_REGION", "us-east-1"),
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"), # Use IAM roles in production!
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"), # Use IAM roles in production!
        "mcp_server_url": os.getenv("MCP_SERVER_URL"),
        # Add default model IDs if needed
        "default_ollama_model": os.getenv("DEFAULT_OLLAMA_MODEL", "llama3"),
        "default_bedrock_model": os.getenv("DEFAULT_BEDROCK_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0"),
    }
