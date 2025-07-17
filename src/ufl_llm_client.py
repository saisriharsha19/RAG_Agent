import aiohttp
import asyncio
import json
import re
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

# Use your existing environment variables
UFL_AI_BASE_URL = os.getenv("UFL_AI_BASE_URL")
UFL_AI_MODEL = os.getenv("UFL_AI_MODEL")
UFL_AI_API_KEY = os.getenv("UFL_AI_API_KEY")

def extract_json_from_text(text):
    """Extract JSON from text response"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    json_patterns = [
        r'\{.*\}',
        r'\[.*\]'
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    return None

class UFL_LLMClient:
    """Simple UFL client without global session pooling"""
    
    def __init__(self):
        self.base_url = UFL_AI_BASE_URL
        self.model = UFL_AI_MODEL
        self.api_key = UFL_AI_API_KEY
        
        # Validate configuration
        if not self.base_url:
            logger.warning("UFL_AI_BASE_URL not found in environment variables")
            raise ValueError("UFL_AI_BASE_URL is required but not set")
        
        if not self.model:
            logger.warning("UFL_AI_MODEL not found in environment variables")
            raise ValueError("UFL_AI_MODEL is required but not set")
            
        if not self.api_key:
            logger.warning("UFL_AI_API_KEY not found in environment variables")
            raise ValueError("UFL_AI_API_KEY is required but not set")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def call_ufl_api(self, prompt: str, use_json_format: bool = False) -> Dict[str, Any]:
        """Call UFL AI API with fresh session each time"""
        
        # Create fresh session for each request
        async with aiohttp.ClientSession() as session:
            try:
                logger.info(f"Calling UFL AI API")
                logger.debug(f"Prompt: {prompt[:200]}...")
                
                # Headers
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                # Request data
                data = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}]
                }
                
                if use_json_format:
                    data["response_format"] = {"type": "json_object"}
                
                # Debug logging
                logger.debug(f"Making request to: {self.base_url}/chat/completions")
                logger.debug(f"Headers: {headers}")
                logger.debug(f"Data: {data}")
                
                # Make the API call
                async with session.post(
                    f"{self.base_url}/chat/completions", 
                    json=data, 
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    # Check response status
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"API returned status {response.status}: {error_text}")
                        raise Exception(f"API returned status {response.status}: {error_text}")
                    
                    # Parse response
                    result = await response.json()
                    logger.debug(f"API response: {result}")
                    
                    # Extract content
                    if "choices" not in result or not result["choices"]:
                        raise Exception("Invalid response format - no choices")
                    
                    content = result["choices"][0]["message"]["content"]
                    logger.info(f"Successfully received response, length: {len(content)} characters")
                    
                    # Handle JSON format
                    if use_json_format:
                        parsed_content = extract_json_from_text(content)
                        if not parsed_content:
                            logger.error(f"Failed to parse response as JSON: {content[:500]}")
                            return {"error": "Invalid JSON response", "content": content}
                        return parsed_content
                    else:
                        return {"content": content}
                        
            except aiohttp.ClientConnectorError as e:
                logger.error(f"Connection error: {e}")
                raise Exception(f"Connection error - check UFL_AI_BASE_URL: {e}")
            except aiohttp.ServerTimeoutError:
                logger.error("Request timeout")
                raise Exception("Request timeout - API took too long to respond")
            except aiohttp.ClientResponseError as e:
                logger.error(f"HTTP error: {e}")
                raise Exception(f"HTTP error {e.status}: {e.message}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                raise Exception("Invalid JSON response from API")
            except Exception as e:
                logger.error(f"API request failed: {str(e)}")
                raise Exception(f"API request failed: {str(e)}")
    
    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text response"""
        
        try:
            # Combine prompts
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            
            logger.info(f"Generating response for prompt length: {len(full_prompt)} characters")
            
            # Make API call
            result = await self.call_ufl_api(full_prompt, use_json_format=False)
            
            # Extract content
            content = result.get("content", "")
            if not content:
                return "Error: Empty response from API"
            
            logger.info(f"Response generated successfully, length: {len(content)} characters")
            return content
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def generate_json_response(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate JSON response"""
        
        try:
            # Combine prompts
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant: I'll respond in JSON format."
            
            # Make API call
            result = await self.call_ufl_api(full_prompt, use_json_format=True)
            return result
            
        except Exception as e:
            logger.error(f"JSON generation failed: {str(e)}")
            return {"error": f"JSON generation failed: {str(e)}"}
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test the API connection"""
        try:
            logger.info("Testing UFL API connection...")
            test_response = await self.generate_response("Hello, this is a test. Please respond with 'Test successful.'")
            
            if "Error:" in test_response:
                return {
                    "status": "failed",
                    "error": test_response,
                    "model": self.model,
                    "endpoint": self.base_url
                }
            else:
                logger.info("API connection test successful")
                return {
                    "status": "success",
                    "response": test_response,
                    "model": self.model,
                    "endpoint": self.base_url
                }
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "model": self.model,
                "endpoint": self.base_url
            }

class SimpleBackupClient:
    """Simple backup client when UFL AI is not available"""
    
    def __init__(self):
        self.model = "backup-client"
        logger.info("Using backup LLM client - UFL AI not configured")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate backup response"""
        await asyncio.sleep(0.1)
        
        prompt_lower = prompt.lower()
        
        if "hello" in prompt_lower or "hi" in prompt_lower:
            return "Hello! I'm currently using a backup client. Please configure your UFL AI settings for full functionality."
        
        if "test" in prompt_lower:
            return "Test successful - backup client is working. Please configure UFL AI for full functionality."
        
        return f"I'm currently using a backup client and cannot provide a detailed response. Please configure your UFL AI settings for full functionality."
    
    async def generate_json_response(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate backup JSON response"""
        return {
            "status": "backup_mode",
            "message": "UFL AI not configured - using backup client",
            "prompt_received": prompt[:100]
        }
    
    async def test_connection(self) -> Dict[str, Any]:
        return {
            "status": "backup",
            "message": "Using backup client - UFL AI not configured",
            "model": self.model
        }

def create_llm_client():
    """Factory function to create simple UFL client"""
    try:
        client = UFL_LLMClient()
        logger.info("Simple UFL client created successfully")
        return client
    except ValueError as ve:
        logger.warning(f"UFL AI configuration issue: {ve}")
        return SimpleBackupClient()
    except Exception as e:
        logger.error(f"Unexpected error creating UFL client: {e}")
        return SimpleBackupClient()

# Simple test function
async def test_simple_client():
    """Test the simple client"""
    try:
        client = create_llm_client()
        async with client:
            response = await client.generate_response("Hello, say 'Simple client working!'")
        print(f"Test response: {response}")
        return "Error:" not in response
    except Exception as e:
        print(f"Test failed: {e}")
        return False