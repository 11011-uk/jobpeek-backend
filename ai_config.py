# AI Model Configurations
AI_MODELS = [
    {
        "name": "Mistral Small 3.2",
        "provider": "OpenRouter",
        "model_id": "mistralai/mistral-small-3.2-24b-instruct:free",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "max_tokens": 3000,
        "temperature": 0.2,
        "cost_per_1k_tokens": 0.0,  # Free  tier
        "requests_per_minute": 30,
        "priority": 1,
        "is_active": True
    },
    {
        "name": "Mistral Small 3.1",
        "provider": "OpenRouter",
        "model_id": "mistralai/mistral-small-3.1-24b-instruct:free",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "max_tokens": 3000,
        "temperature": 0.2,
        "cost_per_1k_tokens": 0.0,  # Free  tier
        "requests_per_minute": 25,
        "priority": 2,
        "is_active": True
    },
    {
        "name": "Llama 3.3 70B",
        "provider": "OpenRouter",
        "model_id": "meta-llama/llama-3.3-70b-instruct:free",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "max_tokens": 3000,
        "temperature": 0.2,
        "cost_per_1k_tokens": 0.0,  # Free tier
        "requests_per_minute": 10,
        "priority": 3,
        "is_active": True
    },
]

# Job Description Formatting Prompt
JOB_FORMATTING_PROMPT = """Reformat this job posting. Start with the job title on its own line, followed by these sections: Job Overview (include company name and location in the overview), Key Benefits, Qualifications, Responsibilities.

Title: {title}
Company: {company}
Location: {location}
Description:
{description}
"""

# System Message for AI Models
SYSTEM_MESSAGE = "You are an expert HR assistant. Keep outputs concise, professional, and well-structured. Always include the company name and location in the Job Overview section."

# Rate Limiting Settings
DEFAULT_RATE_LIMIT = 25  # requests per minute
RATE_LIMIT_BUFFER = 0.99  # Use 80% of actual limit to be safe

# Cost Tracking
ENABLE_COST_TRACKING = True
COST_ALERT_THRESHOLD = 0.10  # Alert when cost exceeds $0.10

# Fallback Settings
ENABLE_AUTO_FALLBACK = True
MAX_RETRIES_PER_MODEL = 2
RETRY_DELAY = 1.0  # seconds

# Logging Settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "[{asctime}] {levelname}: {message}"
LOG_FILE = "ai_service.log"

# Performance Settings
REQUEST_TIMEOUT = 30  # seconds
BATCH_SIZE = 5  # Number of jobs to process in parallel (if implementing async)
