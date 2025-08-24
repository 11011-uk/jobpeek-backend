import os
import time
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv

load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AIModel:
    """AI Model Configuration"""
    name: str
    provider: str
    model_id: str
    api_key_env: str
    base_url: str
    max_tokens: int
    temperature: float
    cost_per_1k_tokens: float
    requests_per_minute: int
    is_active: bool = True
    priority: int = 1  # Lower number = higher priority

class RateLimiter:
    """Rate limiting per model"""
    
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.requests: List[float] = []
    
    def can_make_request(self) -> bool:
        """Check if we can make a request now"""
        now = time.time()
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        return len(self.requests) < self.requests_per_minute
    
    def record_request(self):
        """Record a request"""
        self.requests.append(time.time())
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        if not self.can_make_request():
            wait_time = 60 - (time.time() - self.requests[0]) + 1
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)

class AIService:
    """AI Service with multiple model support and fallback"""
    
    def __init__(self):
        self.models: List[AIModel] = []
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.model_clients: Dict[str, any] = {}
        self.initialize_models()
        self.initialize_clients()
    
    def initialize_models(self):
        """Initialize available AI models"""
        
        # Primary Model (OpenRouter + Mistral)
        self.models.append(AIModel(
            name="Mistral Small 3.2",
            provider="OpenRouter",
            model_id="mistralai/mistral-small-3.2-24b-instruct:free",
            api_key_env="OPENAI_API_KEY",
            base_url="https://openrouter.ai/api/v1",
            max_tokens=3000,
            temperature=0.2,
            cost_per_1k_tokens=0.0,  # Free tier
            requests_per_minute=30,
            priority=1
        ))
        
        # Alternative OpenRouter Model (if available)
        self.models.append(AIModel(
            name="Mistral Small 3.1",
            provider="OpenRouter",
            model_id="mistralai/mistral-small-3.1-24b-instruct:free",
            api_key_env="OPENAI_API_KEY",
            base_url="https://openrouter.ai/api/v1",
            max_tokens=3000,
            temperature=0.2,
            cost_per_1k_tokens=0.0,  # Free tier
            requests_per_minute=25,
            priority=2
        ))
        
        # Initialize rate limiters
        for model in self.models:
            self.rate_limiters[model.name] = RateLimiter(model.requests_per_minute)
    
    def initialize_clients(self):
        """Initialize API clients for each model"""
        try:
            from openai import OpenAI
            for model in self.models:
                if model.provider in ["OpenRouter", "OpenAI"]:
                    api_key = os.getenv(model.api_key_env)
                    if api_key:
                        self.model_clients[model.name] = OpenAI(
                            api_key=api_key,
                            base_url=model.base_url
                        )
                        logger.info(f"✅ Initialized client for {model.name}")
        except ImportError:
            logger.warning("OpenAI client not available")
    
    def get_available_models(self) -> List[AIModel]:
        """Get list of available models sorted by priority"""
        return sorted(
            [model for model in self.models if model.is_active and model.name in self.model_clients],
            key=lambda x: x.priority
        )
    
    def format_job_description(self, title: str, company: str, location: str, description: str) -> Tuple[str, str, float]:
        """
        Format job description using available AI models
        Returns: (formatted_text, model_used, cost_incurred)
        """
        
        prompt = f"""Reformat this job posting. Start with the job title on its own line, followed by these sections: Job Overview (include company name and location in the overview), Key Benefits, Qualifications, Responsibilities.

Title: {title or ''}
Company: {company or ''}
Location: {location or ''}
Description:
{description or ''}
"""
        
        available_models = self.get_available_models()
        
        for model in available_models:
            try:
                # Check rate limiting
                rate_limiter = self.rate_limiters[model.name]
                if not rate_limiter.can_make_request():
                    logger.info(f"Rate limit reached for {model.name}, trying next model")
                    continue
                
                logger.info(f"🔄 Attempting to format with {model.name}")
                
                # Make the request
                rate_limiter.record_request()
                formatted_text = self._call_model(model, prompt)
                
                if formatted_text:
                    # Calculate cost
                    estimated_tokens = len(formatted_text.split()) * 1.3  # Rough estimation
                    cost = (estimated_tokens / 1000) * model.cost_per_1k_tokens
                    
                    logger.info(f"✅ Successfully formatted with {model.name} (Cost: ${cost:.4f})")
                    return formatted_text, model.name, cost
                
            except Exception as e:
                logger.error(f"❌ Error with {model.name}: {str(e)}")
                continue
        
        # If all models fail, return original description
        logger.error("❌ All AI models failed, returning original description")
        return description, "None", 0.0
    
    def _call_model(self, model: AIModel, prompt: str) -> Optional[str]:
        """Call specific AI model"""
        
        try:
            if model.provider in ["OpenRouter", "OpenAI"]:
                client = self.model_clients[model.name]
                response = client.chat.completions.create(
                    model=model.model_id,
                    messages=[
                        {"role": "system", "content": "You are an expert HR assistant. Keep outputs concise."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=model.temperature,
                    max_tokens=model.max_tokens
                )
                return response.choices[0].message.content.strip()
            
            elif model.provider == "Anthropic":
                client = self.model_clients[model.name]
                response = client.messages.create(
                    model=model.model_id,
                    max_tokens=model.max_tokens,
                    temperature=model.temperature,
                    system="You are an expert HR assistant. Keep outputs concise.",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()
                
        except Exception as e:
            logger.error(f"Error calling {model.name}: {str(e)}")
            return None
    
    def get_model_status(self) -> Dict[str, Dict]:
        """Get status of all models"""
        status = {}
        for model in self.models:
            rate_limiter = self.rate_limiters[model.name]
            status[model.name] = {
                "provider": model.provider,
                "is_active": model.is_active,
                "priority": model.priority,
                "requests_available": rate_limiter.requests_per_minute - len(rate_limiter.requests),
                "cost_per_1k_tokens": model.cost_per_1k_tokens,
                "client_available": model.name in self.model_clients
            }
        return status
    
    def add_custom_model(self, model_config: AIModel):
        """Add a custom AI model"""
        self.models.append(model_config)
        self.rate_limiters[model_config.name] = RateLimiter(model_config.requests_per_minute)
        logger.info(f"✅ Added custom model: {model_config.name}")
    
    def disable_model(self, model_name: str):
        """Disable a specific model"""
        for model in self.models:
            if model.name == model_name:
                model.is_active = False
                logger.info(f"✅ Disabled model: {model_name}")
                break
    
    def enable_model(self, model_name: str):
        """Enable a specific model"""
        for model in self.models:
            if model.name == model_name:
                model.is_active = True
                logger.info(f"✅ Enabled model: {model_name}")
                break

# Global AI service instance
ai_service = AIService()

# Convenience functions for backward compatibility
def call_gpt_format(title: str, company: str, location: str, description: str) -> str:
    """Backward compatibility function"""
    formatted_text, model_used, cost = ai_service.format_job_description(title, company, location, description)
    return formatted_text

def get_ai_status() -> Dict[str, Dict]:
    """Get AI service status"""
    return ai_service.get_model_status()

if __name__ == "__main__":
    # Test the AI service
    print("🤖 AI Service Test")
    print("=" * 50)
    
    # Show available models
    print("\n📋 Available Models:")
    for model in ai_service.get_available_models():
        print(f"  • {model.name} ({model.provider}) - Priority: {model.priority}")
    
    # Show model status
    print("\n📊 Model Status:")
    status = ai_service.get_model_status()
    for model_name, info in status.items():
        print(f"  • {model_name}: {'✅' if info['client_available'] else '❌'} - Requests: {info['requests_available']}")
    
    # Test formatting
    print("\n🧪 Testing Formatting:")
    test_result = call_gpt_format(
        "Software Engineer",
        "Tech Corp",
        "San Francisco, CA",
        "We are looking for a talented software engineer..."
    )
    print(f"Result: {test_result[:100]}...")
