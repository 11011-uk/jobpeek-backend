import os
import re
import datetime
import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import redis
import time
import logging
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis connection with retry logic
def create_redis_connection():
    """Create Redis connection with retry logic for Redis Cloud"""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            # Get Redis connection details
            redis_url = os.environ.get("REDIS_URL")
            if not redis_url:
                raise ValueError("REDIS_URL environment variable is required")
            
            logger.info(f"Attempting Redis connection (attempt {attempt + 1}/{max_retries})")
            
            # Parse Redis URL for connection details
            parsed_url = urlparse(redis_url)
            
            # Create Redis connection with SSL support for Redis Cloud
            r = redis.Redis(
                host=parsed_url.hostname,
                port=parsed_url.port or 6379,
                password=parsed_url.password,
                decode_responses=True,
                ssl=parsed_url.scheme == 'rediss',  # Enable SSL for rediss:// URLs
                ssl_cert_reqs=None,  # Disable certificate verification for Redis Cloud
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            r.ping()
            logger.info("Redis connection successful")
            return r
            
        except Exception as e:
            logger.error(f"Redis connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error("All Redis connection attempts failed")
                raise

# Initialize Redis with retry logic
r = create_redis_connection()

def redis_retry_wrapper(func, *args, **kwargs):
    """Wrapper to retry Redis operations on connection failure"""
    global r
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Redis operation failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                # Try to reconnect
                try:
                    r = create_redis_connection()
                except:
                    time.sleep(1)
            else:
                logger.error("All Redis retry attempts failed")
                raise HTTPException(status_code=503, detail="Redis service temporarily unavailable")

BASE_URL = os.getenv("BASE_URL", "https://us.careerdays.io")
app = FastAPI(title="JobPeek API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def today_keys(user_id: str):
    """Generate Redis keys for today's data"""
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    seen_key = f"seen:{date_str}:{user_id}"
    history_key = f"history:{date_str}:{user_id}"
    return seen_key, history_key

def cache_get(url: str) -> str:
    """Get cached HTML or fetch and cache it"""
    cache_key = f"cache:page:{url}"
    
    # Try to get from cache first
    try:
        cached_html = redis_retry_wrapper(r.get, cache_key)
        if cached_html:
            logger.info(f"Cache hit for {url}")
            return cached_html
    except:
        logger.warning(f"Cache lookup failed for {url}, fetching directly")
    
    # Fetch from source
    logger.info(f"Fetching {url}")
    try:
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            # Add some basic headers to look like a real browser
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = client.get(url, headers=headers)
            response.raise_for_status()
            html = response.text
        
        # Try to cache for 90 seconds
        try:
            redis_retry_wrapper(r.setex, cache_key, 90, html)
            logger.info(f"Cached {url} for 90 seconds")
        except:
            logger.warning(f"Failed to cache {url}, continuing without cache")
        
        # Be polite - add a small delay
        time.sleep(0.5)
        
        return html
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch job data: {str(e)}")

def discover_job_ids() -> list:
    """Discover job IDs from the search page"""
    search_url = f"{BASE_URL}/search/all?platform=ss&page=1"
    html = cache_get(search_url)
    
    # Extract job IDs using regex pattern from PRD
    job_ids = re.findall(r'/jobs/[^"]*?-(\w{24})-ss', html)
    logger.info(f"Discovered {len(job_ids)} job IDs")
    return job_ids

def parse_job_details(job_id: str) -> dict:
    """Parse job details from job page"""
    # Use a generic job title in URL since the actual title doesn't matter
    job_url = f"{BASE_URL}/jobs/job-{job_id}-ss"
    html = cache_get(job_url)
    
    try:
        soup = BeautifulSoup(html, "html.parser")
        
        # Extract title (first h1 tag)
        title_elem = soup.find("h1")
        title = title_elem.get_text(strip=True) if title_elem else "Job Title Not Found"
        
        # Extract company name - try different selectors
        company = "Company Not Found"
        company_selectors = [
            ".company-name",
            "[data-testid='company-name']",
            ".employer",
            "h2",
            ".job-company"
        ]
        
        for selector in company_selectors:
            company_elem = soup.select_one(selector)
            if company_elem:
                company = company_elem.get_text(strip=True)
                break
        
        # Extract job description (article tag or main content)
        description_html = "<p>Job description not available</p>"
        desc_selectors = [
            "article",
            ".job-description",
            ".description",
            "[data-testid='job-description']",
            ".content"
        ]
        
        for selector in desc_selectors:
            desc_elem = soup.select_one(selector)
            if desc_elem:
                description_html = str(desc_elem)
                break
        
        return {
            "id": job_id,
            "title": title,
            "company": company,
            "description_html": description_html,
            "original_url": job_url
        }
        
    except Exception as e:
        logger.error(f"Failed to parse job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse job details: {str(e)}")

def set_daily_expiry(key: str):
    """Set key to expire at midnight"""
    now = datetime.datetime.now()
    midnight = (now + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    seconds_until_midnight = int((midnight - now).total_seconds())
    try:
        redis_retry_wrapper(r.expire, key, seconds_until_midnight)
    except:
        logger.warning(f"Failed to set expiry on key {key}")

@app.get("/")
async def root():
    """Health check endpoint"""
    try:
        redis_retry_wrapper(r.ping)
        redis_status = "connected"
    except:
        redis_status = "disconnected"
    
    return {
        "message": "JobPeek API is running", 
        "status": "healthy",
        "redis_status": redis_status
    }

@app.get("/api/next")
async def get_next_job(user: str = Query(..., description="User ID")):
    """Get the next unseen job for a user"""
    if not user:
        raise HTTPException(status_code=400, detail="User ID is required")
    
    seen_key, history_key = today_keys(user)
    
    try:
        # Discover available job IDs
        job_ids = discover_job_ids()
        
        if not job_ids:
            return {"error": "No jobs found on the site"}
        
        # Find first unseen job
        for job_id in job_ids:
            try:
                is_seen = redis_retry_wrapper(r.sismember, seen_key, job_id)
                if not is_seen:
                    # Parse job details
                    job_data = parse_job_details(job_id)
                    
                    # Mark as seen and add to history
                    redis_retry_wrapper(r.sadd, seen_key, job_id)
                    redis_retry_wrapper(r.rpush, history_key, job_id)
                    
                    # Set daily expiry on keys
                    set_daily_expiry(seen_key)
                    set_daily_expiry(history_key)
                    
                    logger.info(f"Served new job {job_id} to user {user}")
                    return job_data
            except Exception as redis_error:
                logger.warning(f"Redis error for job {job_id}: {redis_error}, continuing...")
                continue
        
        return {"error": "No new jobs found - you've seen all available jobs today"}
        
    except Exception as e:
        logger.error(f"Error in get_next_job: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/prev")
async def get_previous_job(user: str = Query(..., description="User ID")):
    """Get the previous job from user's history"""
    if not user:
        raise HTTPException(status_code=400, detail="User ID is required")
    
    seen_key, history_key = today_keys(user)
    
    try:
        # Get history length
        history_length = redis_retry_wrapper(r.llen, history_key)
        
        if history_length < 2:
            return {"error": "No previous job available"}
        
        # Get second-to-last job (last is current)
        prev_index = history_length - 2
        job_id = redis_retry_wrapper(r.lindex, history_key, prev_index)
        
        if not job_id:
            return {"error": "No previous job found"}
        
        # Parse and return job details
        job_data = parse_job_details(job_id)
        logger.info(f"Served previous job {job_id} to user {user}")
        return job_data
        
    except Exception as e:
        logger.error(f"Error in get_previous_job: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/stats")
async def get_user_stats(user: str = Query(..., description="User ID")):
    """Get user statistics for debugging"""
    seen_key, history_key = today_keys(user)
    
    try:
        jobs_seen = redis_retry_wrapper(r.scard, seen_key)
        history_len = redis_retry_wrapper(r.llen, history_key)
    except:
        jobs_seen = 0
        history_len = 0
    
    return {
        "user_id": user,
        "jobs_seen_today": jobs_seen,
        "history_length": history_len,
        "date": datetime.datetime.now().strftime("%Y-%m-%d")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
