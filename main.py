import os
import sqlite3
import time
import hashlib
import html
import re
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
load_dotenv(override=True)

# Import the new AI service
from ai_service import ai_service, call_gpt_format

# optional html -> text helper
try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except Exception:
    _HAS_BS4 = False

DB_PATH = "jobs.db"

app = FastAPI(title="Job Scraper API", version="3.0.0 (Multi-AI Service)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)  # increased timeout for concurrent writes
    conn.row_factory = sqlite3.Row
    return conn

def row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return dict(row)

def clean_html_to_text(html_text: Optional[str]) -> str:
    if not html_text:
        return ""
    s = html.unescape(html_text)
    if _HAS_BS4:
        return BeautifulSoup(s, "html.parser").get_text(separator="\n", strip=True)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\s+\n", "\n", s)
    return s.strip()

# Ensure schema: add columns if missing
def ensure_schema():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(jobs)")
    cols = [r["name"] for r in cur.fetchall()]
    to_add = []
    if "formatted_description" not in cols:
        to_add.append("ALTER TABLE jobs ADD COLUMN formatted_description TEXT;")
    if "formatted_at" not in cols:
        to_add.append("ALTER TABLE jobs ADD COLUMN formatted_at TEXT;")
    if "formatted_model" not in cols:
        to_add.append("ALTER TABLE jobs ADD COLUMN formatted_model TEXT;")
    if "hidden" not in cols:
        to_add.append("ALTER TABLE jobs ADD COLUMN hidden INTEGER DEFAULT 0;")
    for sql in to_add:
        try:
            cur.execute(sql)
        except Exception as e:
            print(f"[schema] warning: {e}")
    conn.commit()
    conn.close()

ensure_schema()

# Updated GPT call using AI service
def call_gpt_format_with_service(title: str, company: str, location: str, raw_description: str) -> str:
    base = clean_html_to_text(raw_description or "")
    if not base:
        print(f"[AI] No description text to format")
        return ""
    
    print(f"[AI] Formatting job: {title[:50]}... at {company}")
    
    try:
        formatted_text, model_used, cost = ai_service.format_job_description(title, company, location, base)
        print(f"[AI] Successfully formatted with {model_used} (Cost: ${cost:.4f})")
        return formatted_text
    except Exception as e:
        print(f"[AI] Error in AI service: {e}")
        return base

# helper: persist formatted text into DB (with small retry loop for 'database is locked')
def save_formatted_to_db(entity_id: str, formatted_text: str, model_name: str = "multi-ai-service"):
    attempts = 0
    while attempts < 3:
        attempts += 1
        try:
            conn = get_db_conn()
            cur = conn.cursor()
            cur.execute(
                "UPDATE jobs SET formatted_description = ?, formatted_at = ?, formatted_model = ? WHERE entity_id = ?",
                (formatted_text, datetime.utcnow().isoformat() + "Z", model_name, entity_id)
            )
            conn.commit()
            conn.close()
            print(f"[DB] Successfully saved formatted description for {entity_id}")
            return True
        except sqlite3.OperationalError as e:
            print(f"[DB] write attempt {attempts} failed: {e}. retrying...")
            time.sleep(0.5 * attempts)
        except Exception as e:
            print(f"[DB] unexpected save error: {e}")
            return False
    
    print(f"[DB] Failed to save formatted description after 3 attempts for {entity_id}")
    return False

def ensure_url_field(job_dict: Dict[str, Any]):
    # ensure both apply_url and url fields exist and are consistent
    if "apply_url" in job_dict:
        job_dict["url"] = job_dict["apply_url"]
    elif "url" in job_dict:
        job_dict["apply_url"] = job_dict["url"]
    return job_dict

# ------------------ API endpoints ------------------

@app.get("/jobs/count")
def get_jobs_count():
    """Get total count of non-hidden jobs in the database"""
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM jobs WHERE hidden = 0")
    count = cur.fetchone()[0]
    conn.close()
    return {"total": count}

@app.get("/jobs")
def list_jobs(page: int = Query(1, ge=1), limit: int = Query(50, le=200)):
    offset = (page - 1) * limit
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM jobs WHERE hidden = 0 ORDER BY rowid LIMIT ? OFFSET ?", (limit, offset))
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        d = row_to_dict(r)
        ensure_url_field(d)
        
        # If formatted_description exists, use it for the description field
        if d.get("formatted_description"):
            d["description"] = d["formatted_description"]
        
        out.append(d)
    return out

@app.get("/jobs/{entity_id}")
def get_job(entity_id: str):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM jobs WHERE entity_id = ?", (entity_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    job = row_to_dict(row)
    ensure_url_field(job)

    # if formatted_description exists, return it; otherwise format, persist, and return
    if job.get("formatted_description"):
        job["description"] = job["formatted_description"]
        return job

    # synchronous formatting (blocks until formatted or fallback)
    formatted = call_gpt_format_with_service(job.get("title",""), job.get("company",""), job.get("location",""), job.get("description",""))
    # attempt to persist (best-effort)
    saved = save_formatted_to_db(entity_id, formatted)
    # return formatted (whether saved or not) AND update the job object
    job["description"] = formatted
    job["formatted_description"] = formatted  # CRITICAL FIX: Update the field in the job object
    return job

@app.get("/jobs/next/{current_id}")
def get_next_job(current_id: str):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT rowid FROM jobs WHERE entity_id = ?", (current_id,))
    r = cur.fetchone()
    if not r:
        conn.close()
        raise HTTPException(status_code=404, detail="Current job not found")
    cur.execute("SELECT * FROM jobs WHERE rowid > ? AND hidden = 0 ORDER BY rowid ASC LIMIT 1", (r["rowid"],))
    n = cur.fetchone()
    conn.close()
    if not n:
        raise HTTPException(status_code=404, detail="No more jobs available")
    job = row_to_dict(n)
    ensure_url_field(job)

    if job.get("formatted_description"):
        job["description"] = job["formatted_description"]
        return job

    formatted = call_gpt_format_with_service(job.get("title",""), job.get("company",""), job.get("location",""), job.get("description",""))
    save_formatted_to_db(job.get("entity_id"), formatted)
    job["description"] = formatted
    job["formatted_description"] = formatted  # CRITICAL FIX: Update the field in the job object
    return job

@app.get("/jobs/prev/{current_id}")
def get_prev_job(current_id: str):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT rowid FROM jobs WHERE entity_id = ?", (current_id,))
    r = cur.fetchone()
    if not r:
        conn.close()
        raise HTTPException(status_code=404, detail="Current job not found")
    cur.execute("SELECT * FROM jobs WHERE rowid < ? AND hidden = 0 ORDER BY rowid DESC LIMIT 1", (r["rowid"],))
    p = cur.fetchone()
    conn.close()
    if not p:
        raise HTTPException(status_code=404, detail="No previous job available")
    job = row_to_dict(p)
    ensure_url_field(job)

    if job.get("formatted_description"):
        job["description"] = job["formatted_description"]
        return job

    formatted = call_gpt_format_with_service(job.get("title",""), job.get("company",""), job.get("location",""), job.get("description",""))
    save_formatted_to_db(job.get("entity_id"), formatted)
    job["description"] = formatted
    job["formatted_description"] = formatted  # CRITICAL FIX: Update the field in the job object
    return job

@app.get("/ai/status")
def get_ai_status():
    """Get AI service status and model information"""
    return ai_service.get_model_status()

# serve static files (unchanged)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
