# /core/models.py
from abc import ABC, abstractmethod
import ollama
import boto3
import json
import os
import time
import hashlib
from typing import Optional, Dict, Any, Tuple, List  # Added List import here
import tiktoken  # For token counting with OpenAI-compatible models
from langchain_core.language_models.llms import LLM
from langchain_community.llms import Ollama as LangchainOllama
from langchain_aws import BedrockLLM
import random
from pathlib import Path

# Persistent Caching

class PersistentCache:
    def __init__(self, cache_dir="./cache", max_age_days=7, memory_cache_size=100):
        """
        Initialize a persistent cache with both memory and disk storage.
        
        Args:
            cache_dir: Directory to store cache files
            max_age_days: Maximum age of cached items in days
            memory_cache_size: Maximum number of items to keep in memory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_age_seconds = max_age_days * 24 * 60 * 60
        self.memory_cache = {}
        self.memory_cache_size = memory_cache_size
        
        # Load existing cache files into memory (up to memory_cache_size)
        self._preload_cache()
        
    def _preload_cache(self):
        """Load most recent cache files into memory up to memory_cache_size."""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            # Sort by modification time (most recent first)
            cache_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Load up to memory_cache_size files
            count = 0
            for cache_file in cache_files:
                if count >= self.memory_cache_size:
                    break
                    
                # Check if cache is expired
                if time.time() - cache_file.stat().st_mtime > self.max_age_seconds:
                    cache_file.unlink()  # Delete expired cache
                    continue
                    
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                        if 'key' in cached_data and 'value' in cached_data:
                            self.memory_cache[cached_data['key']] = cached_data['value']
                            count += 1
                except Exception as e:
                    print(f"Error loading cache file {cache_file}: {e}")
            
            print(f"Preloaded {count} cache entries into memory.")
        except Exception as e:
            print(f"Error preloading cache: {e}")
        
    def get(self, key):
        """Get a value from the cache (checks memory first, then disk)."""
        # Check memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]
            
        # Check disk cache
        cache_file = self.cache_dir / f"{self._safe_filename(key)}.json"
        if cache_file.exists():
            # Check if cache is expired
            if time.time() - cache_file.stat().st_mtime > self.max_age_seconds:
                cache_file.unlink()  # Delete expired cache
                return None
                
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    
                    # Make sure the key matches (in case of hash collisions)
                    if cached_data.get('key') == key:
                        # Update memory cache (LRU policy - evict if full)
                        if len(self.memory_cache) >= self.memory_cache_size:
                            # Remove a random entry - approximating LRU without tracking access times
                            evict_key = random.choice(list(self.memory_cache.keys()))
                            del self.memory_cache[evict_key]
                            
                        self.memory_cache[key] = cached_data['value']
                        return cached_data['value']
            except Exception as e:
                print(f"Error reading cache file for {key}: {e}")
                
        return None
        
    def set(self, key, value):
        """Set a value in both memory and disk cache."""
        # Update memory cache (with LRU-like eviction)
        if len(self.memory_cache) >= self.memory_cache_size:
            # Remove a random entry for simplicity
            evict_key = random.choice(list(self.memory_cache.keys()))
            del self.memory_cache[evict_key]
            
        self.memory_cache[key] = value
        
        # Update disk cache
        try:
            cache_file = self.cache_dir / f"{self._safe_filename(key)}.json"
            with open(cache_file, 'w') as f:
                json.dump({
                    'key': key,
                    'value': value,
                    'timestamp': time.time()
                }, f)
        except Exception as e:
            print(f"Error writing to cache file for {key}: {e}")
        
        # Periodically clean up old cache files
        if random.random() < 0.05:  # 5% chance to clean up on each set operation
            self._cleanup()
    
    def _cleanup(self):
        """Remove expired cache files from disk."""
        try:
            now = time.time()
            cleanup_count = 0
            for cache_file in self.cache_dir.glob("*.json"):
                if now - cache_file.stat().st_mtime > self.max_age_seconds:
                    cache_file.unlink()
                    cleanup_count += 1
            
            if cleanup_count > 0:
                print(f"Cache cleanup: removed {cleanup_count} expired cache files.")
        except Exception as e:
            print(f"Error during cache cleanup: {e}")
            
    def _safe_filename(self, key):
        """Convert a key to a safe filename using a hash."""
        return hashlib.md5(str(key).encode()).hexdigest()
        
    def clear(self):
        """Clear both memory and disk cache."""
        self.memory_cache.clear()
        
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            print("Cache cleared successfully.")
        except Exception as e:
            print(f"Error clearing cache: {e}")
            
    def create_key(self, prompt, model_id, **kwargs):
        """Create a deterministic cache key from the inputs."""
        key_parts = [prompt, model_id]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        key_str = "||".join(key_parts)
        return key_str  # No need to hash here, we'll hash for filenames separately

# Simple cache implementation
class ModelResponseCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        
    def get(self, key):
        return self.cache.get(key)
        
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            # Simple eviction policy: remove oldest item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value
        
    def create_key(self, prompt, model_id, **kwargs):
        # Create a deterministic cache key from the inputs
        key_parts = [prompt, model_id]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        key_str = "||".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

# Global response cache
# response_cache = ModelResponseCache() 
response_cache = PersistentCache(cache_dir="./cache/responses", max_age_days=30)

class BaseModelConnector(ABC):
    @abstractmethod
    def generate_content(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def get_langchain_llm(self) -> LLM:
        pass
    
    @abstractmethod
    def estimate_cost(self, prompt: str) -> float:
        """Estimate the cost of generating content for the given prompt."""
        pass
    
    @abstractmethod
    def get_complexity_capability(self) -> int:
        """Return a value 1-10 indicating model capability for complex tasks."""
        pass

class OllamaConnector(BaseModelConnector):
    def __init__(self, config):
        self.base_url = config.get("ollama_base_url")
        self.model_id = config.get("ollama_model_id", config.get("default_ollama_model"))
        self.client = ollama.Client(host=self.base_url)
        self._lc_llm = None
        
        # Approximated costs and capabilities based on model
        self.model_info = {
            "llama3": {"cost_per_1k_tokens": 0.0, "complexity_capability": 7},
            "mistral": {"cost_per_1k_tokens": 0.0, "complexity_capability": 6},
            "phi3": {"cost_per_1k_tokens": 0.0, "complexity_capability": 5},
            # Add more models as needed
            "default": {"cost_per_1k_tokens": 0.0, "complexity_capability": 5}
        }

    def generate_content(self, prompt: str, **kwargs) -> str:
        cache_key = response_cache.create_key(prompt, self.model_id, **kwargs)
        cached_response = response_cache.get(cache_key)
        
        if cached_response:
            print(f"Cache hit for Ollama model {self.model_id}")
            return cached_response
            
        try:
            response = self.client.generate(model=self.model_id, prompt=prompt, stream=False)
            result = response['response']
            
            # Cache the response
            response_cache.set(cache_key, result)
            
            return result
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            return f"Error: Could not generate content using Ollama model {self.model_id}."

    def get_langchain_llm(self) -> LLM:
        if not self._lc_llm:
            self._lc_llm = LangchainOllama(
                model=self.model_id,
                base_url=self.base_url
            )
        return self._lc_llm
    
    def estimate_cost(self, prompt: str) -> float:
        # Ollama is free to run locally, but we account for rough token count
        # for comparison purposes
        tokens = len(prompt) / 4  # Very rough estimate of tokens
        model_config = self.model_info.get(self.model_id, self.model_info["default"])
        return model_config["cost_per_1k_tokens"] * (tokens / 1000)
    
    def get_complexity_capability(self) -> int:
        model_config = self.model_info.get(self.model_id, self.model_info["default"])
        return model_config["complexity_capability"]

class BedrockConnector(BaseModelConnector):
    def __init__(self, config):
        self.region = config.get("aws_region")
        self.client = boto3.client(
            'bedrock-runtime',
            region_name=self.region,
            aws_access_key_id=config.get("aws_access_key_id"),
            aws_secret_access_key=config.get("aws_secret_access_key"),
        )
        self.model_id = config.get("bedrock_model_id", config.get("default_bedrock_model"))
        self._lc_llm = None
        
        # NEW: Add inference profile support
        self.inference_profile_arn = config.get("bedrock_inference_profile_arn")
        self.use_inference_profile = (self.inference_profile_arn is not None and 
                                     len(self.inference_profile_arn) > 0)
        
        if "claude-3-5" in self.model_id.lower() and not self.use_inference_profile:
            print(f"WARNING: Claude 3.5 Sonnet model detected ({self.model_id}) but no inference profile ARN provided.")
            print("Claude 3.5 Sonnet requires an inference profile. Direct invocation may fail.")
        
        # Model info with approximate pricing (as of May 2025)
        self.model_info = {
            "anthropic.claude-3-5-sonnet-20241022-v2:0": {
                "cost_per_1k_input_tokens": 0.003,
                "cost_per_1k_output_tokens": 0.015,
                "complexity_capability": 10,
                "family": "anthropic"
            },
            "anthropic.claude-3-sonnet-20240229-v1:0": {
                "cost_per_1k_input_tokens": 0.003,
                "cost_per_1k_output_tokens": 0.015,
                "complexity_capability": 9,
                "family": "anthropic"
            },
            "anthropic.claude-3-haiku-20240307-v1:0": {
                "cost_per_1k_input_tokens": 0.00025,
                "cost_per_1k_output_tokens": 0.00125,
                "complexity_capability": 7,
                "family": "anthropic"
            },
            "meta.llama3-8b-instruct-v1:0": {
                "cost_per_1k_input_tokens": 0.0002,
                "cost_per_1k_output_tokens": 0.0002,
                "complexity_capability": 6,
                "family": "meta"
            },
            "amazon.titan-text-express-v1": {
                "cost_per_1k_input_tokens": 0.0002,
                "cost_per_1k_output_tokens": 0.0002,
                "complexity_capability": 6,
                "family": "amazon"
            },
            # Default values for unknown models
            "default": {
                "cost_per_1k_input_tokens": 0.003,
                "cost_per_1k_output_tokens": 0.015,
                "complexity_capability": 8,
                "family": "unknown"
            }
        }
        
        # Initialize token counter based on model family - FIXED VERSION
        self.token_counter = None
        try:
            # Try to use the tiktoken library with more reliable fallbacks
            import tiktoken
            try:
                # First try to get p50k_base which is more widely available
                self.token_counter = tiktoken.get_encoding("p50k_base")
                print("Successfully loaded p50k_base tokenizer")
            except:
                try:
                    # Try r50k_base as another option
                    self.token_counter = tiktoken.get_encoding("r50k_base")
                    print("Successfully loaded r50k_base tokenizer")
                except:
                    print("Warning: Could not load tiktoken encodings. Using approximation for token counting.")
        except ImportError:
            print("Warning: tiktoken not installed. Using approximation for token counting.")
        
        # Log the model configuration
        model_family = self.model_info.get(self.model_id, self.model_info["default"]).get("family", "unknown")
        print(f"Initialized BedrockConnector with model: {self.model_id}, family: {model_family}")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text, gracefully handling missing tokenizers."""
        if self.token_counter:
            try:
                return len(self.token_counter.encode(text))
            except Exception as e:
                print(f"Warning: Error counting tokens: {e}. Using approximation.")
        
        # Simple approximation: ~4 characters per token for English text
        return len(text) // 4

    def _create_request_body(self, prompt: str, max_tokens: int = 2048) -> str:
        """Create the appropriate request body for the specific model."""
        model_info = self.model_info.get(self.model_id, self.model_info["default"])
        model_family = model_info.get("family", "unknown")
        
        if model_family == "anthropic":
            # Check if this is a newer Claude 3.5+ model that requires the new message format
            if "claude-3-5" in self.model_id.lower() or "claude-3-opus" in self.model_id.lower():
                return json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.999,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                })
            else:
                # Standard Claude 3 Sonnet/Haiku format
                return json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}]
                })
        elif model_family == "meta":
            return json.dumps({
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": 0.7
            })
        elif model_family == "amazon":
            return json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": 0.7,
                    "topP": 0.9
                }
            })
        elif model_family == "cohere":
            return json.dumps({
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7
            })
        else:
            # Generic fallback - this will likely fail but provides a clear error
            print(f"Warning: Unknown model family '{model_family}' for model '{self.model_id}'. Using generic format.")
            return json.dumps({
                "prompt": prompt,
                "max_tokens": max_tokens
            })

    def _parse_response(self, response: dict) -> str:
        """Parse response based on model family."""
        model_info = self.model_info.get(self.model_id, self.model_info["default"])
        model_family = model_info.get("family", "unknown")
        
        response_body = json.loads(response.get('body').read())
        
        if model_family == "anthropic":
            if 'content' in response_body and len(response_body['content']) > 0:
                # Handle both Claude 3.0 and 3.5 response formats
                if isinstance(response_body['content'][0], dict):
                    if 'text' in response_body['content'][0]:
                        # Claude 3.0 format
                        return response_body['content'][0]['text']
                    elif 'type' in response_body['content'][0] and response_body['content'][0]['type'] == 'text':
                        # Claude 3.5 might have this format
                        return response_body['content'][0].get('text', '')
                    elif 'value' in response_body['content'][0]:
                        # Alternative format sometimes seen
                        return response_body['content'][0]['value']
                # For newer Claude models that might have nested content structure
                elif isinstance(response_body['content'], list):
                    # Try to extract text from any format we might encounter
                    extracted_text = []
                    for item in response_body['content']:
                        if isinstance(item, dict):
                            # Extract from dictionary format
                            if 'text' in item:
                                extracted_text.append(item['text'])
                            elif 'value' in item:
                                extracted_text.append(item['value'])
                        elif isinstance(item, str):
                            # Direct string
                            extracted_text.append(item)
                    
                    if extracted_text:
                        return " ".join(extracted_text)
            
            # If we couldn't parse using the standard format, log the response and raise an error
            print(f"Warning: Non-standard Claude response format: {response_body}")
            raise ValueError(f"Could not parse Claude model response: {response_body}")
            
        elif model_family == "meta":
            if 'generation' in response_body:
                return response_body['generation']
            else:
                raise ValueError(f"Unexpected response format from Meta model: {response_body}")
        elif model_family == "amazon":
            if 'results' in response_body and len(response_body['results']) > 0:
                return response_body['results'][0]['outputText']
            else:
                raise ValueError(f"Unexpected response format from Amazon model: {response_body}")
        elif model_family == "cohere":
            if 'generations' in response_body and len(response_body['generations']) > 0:
                return response_body['generations'][0]['text']
            else:
                raise ValueError(f"Unexpected response format from Cohere model: {response_body}")
        else:
            # Generic attempt to extract useful text
            print(f"Warning: Trying to extract content from unknown model family response: {response_body}")
            # Try several common response formats
            for key in ['text', 'content', 'generated_text', 'response', 'output', 'completion']:
                if key in response_body:
                    return response_body[key]
            
            # If we can't find a standard key, return the entire response as string for debugging
            return f"Could not parse response. Raw response: {response_body}"

    def generate_content(self, prompt: str, **kwargs) -> str:
        cache_key = response_cache.create_key(prompt, self.model_id, **kwargs)
        cached_response = response_cache.get(cache_key)
        
        if cached_response:
            print(f"Cache hit for Bedrock model {self.model_id}")
            return cached_response
            
        # Track token usage - FIXED VERSION
        input_tokens = self._count_tokens(prompt)
        
        max_tokens = kwargs.get("max_tokens", 2048)
        
        # Get the appropriate request body for this model
        body = self._create_request_body(prompt, max_tokens)
        
        # Log the request format being used
        model_info = self.model_info.get(self.model_id, self.model_info["default"])
        model_family = model_info.get("family", "unknown")
        print(f"Using request format for {model_family} model family")
        
        try:
            start_time = time.time()
            
            # Use inference profile if available, otherwise use regular invoke_model
            if self.use_inference_profile and "claude-3-5" in self.model_id.lower():
                print(f"Using inference profile: {self.inference_profile_arn}")
                response = self.client.invoke_model_with_response_stream(
                    body=body,
                    modelId=self.model_id,
                    inferenceProfileArn=self.inference_profile_arn,
                    accept="application/json",
                    contentType="application/json"
                )
                
                # Stream handling is different - we need to collect chunks
                response_chunks = []
                for event in response.get('body'):
                    if 'chunk' in event:
                        chunk = event['chunk'].get('bytes')
                        if chunk:
                            response_chunks.append(chunk)
                
                # Combine chunks and parse
                combined_response = b''.join(response_chunks)
                response_data = json.loads(combined_response)
                
                # Create a structure similar to what we'd get from invoke_model
                # so our _parse_response method still works
                class ResponseWrapper:
                    def __init__(self, data):
                        self.data = data
                    def read(self):
                        return json.dumps(self.data).encode()
                
                response = {'body': ResponseWrapper(response_data)}
            else:
                # Regular model invocation
                response = self.client.invoke_model(body=body, modelId=self.model_id)
            
            end_time = time.time()
            
            # Parse the response based on model family
            result = self._parse_response(response)
                
            # Calculate and log metrics - FIXED VERSION
            latency = end_time - start_time
            output_tokens = self._count_tokens(result)
            estimated_cost = (
                (input_tokens / 1000) * model_info["cost_per_1k_input_tokens"] +
                (output_tokens / 1000) * model_info["cost_per_1k_output_tokens"]
            )
            
            print(f"Bedrock ({self.model_id}) metrics:")
            print(f"  - Input tokens: {input_tokens}")
            print(f"  - Output tokens: {output_tokens}")
            print(f"  - Latency: {latency:.2f}s")
            print(f"  - Estimated cost: ${estimated_cost:.6f}")
            
            # Cache the response
            response_cache.set(cache_key, result)
            
            return result
        except Exception as e:
            # Enhanced error message for Claude 3.5 without inference profile
            if "claude-3-5" in self.model_id.lower() and not self.use_inference_profile and "inference profile" in str(e).lower():
                error_msg = (
                    f"Error: Claude 3.5 Sonnet requires an inference profile. "
                    f"Please create an inference profile in the AWS Bedrock console "
                    f"and provide the ARN in your configuration. Error: {str(e)}"
                )
                print(error_msg)
                return error_msg
            
            print(f"Error invoking Bedrock model {self.model_id}: {e}")
            # Return a more detailed error message to help with debugging
            import traceback
            error_details = traceback.format_exc()
            print(f"Detailed error: {error_details}")
            return f"Error: Could not generate content using Bedrock model {self.model_id}. Error: {str(e)}"

    def get_langchain_llm(self) -> LLM:
        if not self._lc_llm:
            self._lc_llm = BedrockLLM(
                client=self.client,
                model_id=self.model_id
            )
        return self._lc_llm
    
    def estimate_cost(self, prompt: str) -> float:
        # FIXED VERSION
        input_tokens = self._count_tokens(prompt)
        estimated_output_tokens = input_tokens * 1.5  # Assume output is ~1.5x input
            
        model_info = self.model_info.get(self.model_id, self.model_info["default"])
        estimated_cost = (
            (input_tokens / 1000) * model_info["cost_per_1k_input_tokens"] +
            (estimated_output_tokens / 1000) * model_info["cost_per_1k_output_tokens"]
        )
        return estimated_cost
    
    def get_complexity_capability(self) -> int:
        model_info = self.model_info.get(self.model_id, self.model_info["default"])
        return model_info["complexity_capability"]
class HybridConnector(BaseModelConnector):
    def __init__(self, config):
        # Create both connectors
        self.ollama = OllamaConnector(config)
        self.bedrock = BedrockConnector(config)
        
        # Determine default strategy
        self.strategy = config.get("hybrid_strategy", "cost-optimized")
        self.complexity_threshold = config.get("complexity_threshold", 7)
        self.cost_threshold = config.get("cost_threshold", 0.01)
        
        # Cache model selection decisions
        self.model_selection_cache = {}
        
        # Track usage for reporting
        self.usage_stats = {
            "ollama_calls": 0,
            "bedrock_calls": 0,
            "bedrock_failures": 0,
            "fallback_events": 0,
            "total_estimated_cost": 0.0
        }
        
        # Error tracking 
        self.errors = []

    def select_model(self, prompt: str, task_type: str = "general") -> Tuple[BaseModelConnector, str]:
        """Selects the most appropriate model based on advanced heuristics."""
        # Use cached decision if available (based on task_type)
        cache_key = task_type
        if cache_key in self.model_selection_cache:
            model_name, connector = self.model_selection_cache[cache_key]
            return connector, model_name
        
        # Check historical performance data
        analytics = ModelAnalytics()
        recommended_model = analytics.get_recommended_model(task_type)
        
        # Get model capabilities and estimated costs
        ollama_capability = self.ollama.get_complexity_capability()
        bedrock_capability = self.bedrock.get_complexity_capability()
        
        # Base complexity on multiple factors (not just length)
        task_complexity_factors = {
            "architecture_reasoning": 2.5,  # Very complex
            "security_analysis": 2.0,       # Complex
            "component_design": 1.8,        # Moderately complex
            "summary": 0.7,                 # Relatively simple
            "general": 1.0                  # Baseline
        }
        
        # Calculate base complexity score 
        base_complexity = min(10, len(prompt) / 500)
        
        # Apply task-specific factor
        task_factor = task_complexity_factors.get(task_type, 1.0)
        complexity_score = min(10, base_complexity * task_factor)
        
        # Check for specific keywords that might indicate high complexity
        complexity_keywords = [
            "distributed", "scalable", "fault-tolerant", "high-availability",
            "security", "encryption", "authentication", "concurrency", "throughput"
        ]
        
        # Increase complexity score based on keywords
        keyword_matches = sum(1 for keyword in complexity_keywords if keyword.lower() in prompt.lower())
        complexity_score = min(10, complexity_score + (keyword_matches * 0.5))
        
        # Estimate costs
        prompt_length = len(prompt)
        ollama_cost = self.ollama.estimate_cost(prompt)
        bedrock_cost = self.bedrock.estimate_cost(prompt)
        
        # Decision factors with weights
        weights = {
            'capability': 0.4,       # How well can the model handle the task
            'cost': 0.3,             # Cost consideration
            'performance': 0.2,      # Historical performance (speed)
            'reliability': 0.1       # Failure rate
        }
        
        # Calculate scores for each model (higher is better)
        ollama_score = (
            weights['capability'] * (ollama_capability / complexity_score if complexity_score > 0 else 1) +
            weights['cost'] * (1.0 - (ollama_cost / (bedrock_cost + 0.00001))) +  # Higher when ollama is cheaper
            weights['performance'] * (1.0 if recommended_model == 'ollama' else 0.5) +
            weights['reliability'] * (1.0 - (self.usage_stats.get('ollama_failures', 0) / 
                                            max(1, self.usage_stats.get('ollama_calls', 1))))
        )
        
        bedrock_score = (
            weights['capability'] * (bedrock_capability / complexity_score if complexity_score > 0 else 1) +
            weights['cost'] * (1.0 - (bedrock_cost / (ollama_cost + 0.00001))) +  # Higher when bedrock is cheaper
            weights['performance'] * (1.0 if recommended_model == 'bedrock' else 0.5) +
            weights['reliability'] * (1.0 - (self.usage_stats.get('bedrock_failures', 0) / 
                                            max(1, self.usage_stats.get('bedrock_calls', 1))))
        )
        
        # Apply strategy-specific adjustments
        if self.strategy == "cost-optimized":
            ollama_score *= 1.5  # Boost Ollama score to prefer it
        elif self.strategy == "quality-optimized":
            bedrock_score *= 1.2  # Boost Bedrock score to prefer it
        
        # Log decision factors for debugging
        print(f"Decision factors for {task_type} task:")
        print(f"  Complexity score: {complexity_score:.2f}/10")
        print(f"  Ollama capability: {ollama_capability}/10")
        print(f"  Bedrock capability: {bedrock_capability}/10")
        print(f"  Ollama score: {ollama_score:.2f}")
        print(f"  Bedrock score: {bedrock_score:.2f}")
        
        # Select the model with the higher score
        selected = ("ollama", self.ollama) if ollama_score >= bedrock_score else ("bedrock", self.bedrock)
        
        # Cache the decision
        self.model_selection_cache[cache_key] = selected
        
        # Record the selection for analytics
        self.selections[task_type] = selected[0]
        
        return selected[1], selected[0]

    def generate_content(self, prompt: str, task_type: str = "general", **kwargs) -> str:
        connector, model_name = self.select_model(prompt, task_type)
        
        print(f"Using {model_name} for task: {task_type}")
        
        # Track metrics
        if model_name == "ollama":
            self.usage_stats["ollama_calls"] += 1
            
        # Generate content with the selected model
        result = None
        try:
            if model_name == "bedrock":
                self.usage_stats["bedrock_calls"] += 1
                result = connector.generate_content(prompt, **kwargs)
                
                # Check if the result indicates an error
                if result and result.startswith("Error:"):
                    raise Exception(f"Bedrock model returned error: {result}")
                    
                # Update cost tracking
                estimated_cost = self.bedrock.estimate_cost(prompt)
                self.usage_stats["total_estimated_cost"] += estimated_cost
                
        except Exception as e:
            # Log the failure and try to fall back to Ollama
            error_msg = f"Error with {model_name} model: {str(e)}"
            print(f"WARNING: {error_msg} - Falling back to Ollama")
            self.errors.append(error_msg)
            
            if model_name == "bedrock":
                self.usage_stats["bedrock_failures"] += 1
                self.usage_stats["fallback_events"] += 1
                
                # Fallback to Ollama
                try:
                    print(f"Attempting fallback to Ollama for task: {task_type}")
                    result = self.ollama.generate_content(prompt, **kwargs)
                    self.usage_stats["ollama_calls"] += 1
                except Exception as fallback_e:
                    # Both models failed
                    second_error = f"Fallback to Ollama also failed: {str(fallback_e)}"
                    print(f"ERROR: {second_error}")
                    self.errors.append(second_error)
                    result = f"Error: Could not generate content with either model. Primary error: {str(e)}, Fallback error: {str(fallback_e)}"
            else:
                # Ollama was the primary model and it failed - try Bedrock as fallback
                try:
                    print(f"Attempting fallback to Bedrock for task: {task_type}")
                    result = self.bedrock.generate_content(prompt, **kwargs)
                    self.usage_stats["bedrock_calls"] += 1
                    self.usage_stats["fallback_events"] += 1
                    
                    estimated_cost = self.bedrock.estimate_cost(prompt)
                    self.usage_stats["total_estimated_cost"] += estimated_cost
                except Exception as fallback_e:
                    # Both models failed
                    second_error = f"Fallback to Bedrock also failed: {str(fallback_e)}"
                    print(f"ERROR: {second_error}")
                    self.errors.append(second_error)
                    result = f"Error: Could not generate content with either model. Primary error: {str(e)}, Fallback error: {str(fallback_e)}"
        
        if result is None:
            result = "Error: Failed to generate content with all available models."
            
        return result

    def get_langchain_llm(self) -> LLM:
        # Default to Bedrock LLM for LangChain integration
        # This is a simplification; in a real implementation, you might 
        # want to make this more sophisticated
        return self.bedrock.get_langchain_llm()
    
    def estimate_cost(self, prompt: str) -> float:
        # This will estimate using the model that would be selected
        connector, _ = self.select_model(prompt)
        return connector.estimate_cost(prompt)
    
    def get_complexity_capability(self) -> int:
        # Return the highest capability among available models
        return max(self.ollama.get_complexity_capability(), self.bedrock.get_complexity_capability())
    
    def get_usage_stats(self) -> Dict[str, Any]:
        return self.usage_stats
    
    def get_errors(self) -> List[str]:
        return self.errors

class ModelFactory:
    @staticmethod
    def get_connector(provider: str, config: dict) -> BaseModelConnector:
        if provider == "ollama":
            return OllamaConnector(config)
        elif provider == "bedrock":
            return BedrockConnector(config)
        elif provider == "hybrid":
            return HybridConnector(config)
        else:
            raise ValueError(f"Unsupported model provider: {provider}")

class ModelAnalytics:
    def __init__(self, db_path="./model_analytics.json"):
        self.db_path = db_path
        self.stats = self._load_stats()
    
    def _load_stats(self):
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r') as f:
                    return json.load(f)
            return {
                "models": {},
                "total_runs": 0,
                "task_performance": {}
            }
        except Exception as e:
            print(f"Error loading model analytics: {e}")
            return {
                "models": {},
                "total_runs": 0,
                "task_performance": {}
            }
    
    def record_generation(self, model_id, task_type, prompt_length, response_length, 
                          latency, estimated_cost=0.0, error=None):
        # Initialize model stats if not present
        if model_id not in self.stats["models"]:
            self.stats["models"][model_id] = {
                "total_calls": 0,
                "success_calls": 0,
                "error_calls": 0,
                "total_latency": 0,
                "total_cost": 0.0,
                "average_response_length": 0,
                "by_task_type": {}
            }
        
        # Update model stats
        model_stats = self.stats["models"][model_id]
        model_stats["total_calls"] += 1
        self.stats["total_runs"] += 1
        
        if error:
            model_stats["error_calls"] += 1
        else:
            model_stats["success_calls"] += 1
            model_stats["total_latency"] += latency
            model_stats["total_cost"] += estimated_cost
            
            # Update running average of response length
            current_avg = model_stats["average_response_length"]
            model_stats["average_response_length"] = (
                (current_avg * (model_stats["success_calls"] - 1) + response_length) / 
                model_stats["success_calls"]
            )
            
            # Update task-specific stats
            if task_type not in model_stats["by_task_type"]:
                model_stats["by_task_type"][task_type] = {
                    "calls": 0,
                    "total_latency": 0,
                    "total_cost": 0.0
                }
            task_stats = model_stats["by_task_type"][task_type]
            task_stats["calls"] += 1
            task_stats["total_latency"] += latency
            task_stats["total_cost"] += estimated_cost
            
            # Update global task performance
            if task_type not in self.stats["task_performance"]:
                self.stats["task_performance"][task_type] = {
                    "best_model": model_id,
                    "best_latency": latency,
                    "models": {}
                }
            
            task_perf = self.stats["task_performance"][task_type]
            if model_id not in task_perf["models"]:
                task_perf["models"][model_id] = {
                    "average_latency": latency,
                    "calls": 1
                }
            else:
                model_perf = task_perf["models"][model_id]
                avg_latency = ((model_perf["average_latency"] * model_perf["calls"]) + latency) / (model_perf["calls"] + 1)
                model_perf["average_latency"] = avg_latency
                model_perf["calls"] += 1
                
                # Update best model if this one is faster
                if avg_latency < task_perf["best_latency"]:
                    task_perf["best_model"] = model_id
                    task_perf["best_latency"] = avg_latency
        
        # Save updated stats
        self._save_stats()
        
    def _save_stats(self):
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            print(f"Error saving model analytics: {e}")
    
    def get_recommended_model(self, task_type):
        if task_type in self.stats["task_performance"]:
            return self.stats["task_performance"][task_type]["best_model"]
        return None