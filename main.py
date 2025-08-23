# main.py
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

# optional html -> text helper
try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except Exception:
    _HAS_BS4 = False

# --- OpenRouter setup (OpenAI-compatible API for free Mistral model) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # set this environment variable
_OPENAI_OK = False
_USE_NEW_OPENAI = False
_openai_client = None

if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        # Use OpenRouter endpoint with OpenAI-compatible client
        _openai_client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url="https://openrouter.ai/api/v1"  # OpenRouter endpoint
        )
        _USE_NEW_OPENAI = True
        _OPENAI_OK = True
        print(f"[GPT] OpenRouter client initialized successfully with Mistral model")
    except Exception as e:
        print(f"[GPT] OpenRouter client failed: {e}")
        try:
            import openai
            openai.api_key = OPENAI_API_KEY
            openai.api_base = "https://openrouter.ai/api/v1"  # OpenRouter endpoint
            _openai_client = openai
            _USE_NEW_OPENAI = False
            _OPENAI_OK = True
            print(f"[GPT] OpenRouter client initialized with classic client")
        except Exception as e2:
            print(f"[GPT] Classic OpenRouter client also failed: {e2}")
            _OPENAI_OK = False
else:
    print("[GPT] No OPENAI_API_KEY found in environment variables")
    _OPENAI_OK = False

DB_PATH = "jobs.db"

app = FastAPI(title="Job Scraper API", version="2.2.0 (OpenRouter + Mistral)")

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
    for sql in to_add:
        try:
            cur.execute(sql)
        except Exception as e:
            print(f"[schema] warning: {e}")
    conn.commit()
    conn.close()

ensure_schema()

# GPT call with retries + simple backoff
def call_gpt_format(title: str, company: str, location: str, raw_description: str, max_retries: int = 3) -> str:
    base = clean_html_to_text(raw_description or "")
    if not base:
        print(f"[GPT] No description text to format")
        return ""
    
    if not _OPENAI_OK:
        print(f"[GPT] OpenRouter not available, returning raw description")
        return base

    print(f"[GPT] Formatting job: {title[:50]}... at {company}")

    prompt = f"""Reformat this job posting. Start with the job title on its own line, followed by these sections: Job Overview, Key Benefits, Qualifications, Responsibilities.

Title: {title or ''}
Company: {company or ''}
Location: {location or ''}
Description:
{base}
"""

    attempt = 0
    backoff = 1.0
    while attempt < max_retries:
        attempt += 1
        try:
            if _USE_NEW_OPENAI:
                resp = _openai_client.chat.completions.create(
                    model="mistralai/mistral-small-3.2-24b-instruct:free",
                    messages=[
                        {"role":"system","content":"You are an expert HR assistant. Keep outputs concise."},
                        {"role":"user","content":prompt}
                    ],
                    temperature=0.2,
                    max_tokens=3000
                )
                content = ""
                try:
                    content = resp.choices[0].message.content
                except Exception:
                    content = getattr(resp.choices[0], "message", {}).get("content", "")
            else:
                resp = _openai_client.ChatCompletion.create(
                    model="mistralai/mistral-small-3.2-24b-instruct:free",
                    messages=[
                        {"role":"system","content":"You are an expert HR assistant. Keep outputs concise."},
                        {"role":"user","content":prompt}
                    ],
                    temperature=0.2,
                    max_tokens=3000
                )
                try:
                    content = resp.choices[0].message.content
                except Exception:
                    content = resp["choices"][0]["message"]["content"]
            if content:
                print(f"[GPT] Successfully formatted job description with Mistral")
                return content.strip()
        except Exception as e:
            print(f"[GPT] attempt {attempt} failed: {e}")
            time.sleep(backoff)
            backoff *= 2
            continue
    
    print(f"[GPT] All attempts failed, returning raw description")
    return base

# helper: persist formatted text into DB (with small retry loop for 'database is locked')
def save_formatted_to_db(entity_id: str, formatted_text: str, model_name: str = "mistral-small-3.2-24b-instruct"):
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

# ------------------ your existing endpoints (unchanged semantics) ------------------

@app.get("/jobs")
def list_jobs(page: int = Query(1, ge=1), limit: int = Query(50, le=200)):
    offset = (page - 1) * limit
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM jobs ORDER BY rowid LIMIT ? OFFSET ?", (limit, offset))
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        d = row_to_dict(r)
        ensure_url_field(d)
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
    formatted = call_gpt_format(job.get("title",""), job.get("company",""), job.get("location",""), job.get("description",""))
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
    cur.execute("SELECT * FROM jobs WHERE rowid > ? ORDER BY rowid ASC LIMIT 1", (r["rowid"],))
    n = cur.fetchone()
    conn.close()
    if not n:
        raise HTTPException(status_code=404, detail="No more jobs available")
    job = row_to_dict(n)
    ensure_url_field(job)

    if job.get("formatted_description"):
        job["description"] = job["formatted_description"]
        return job

    formatted = call_gpt_format(job.get("title",""), job.get("company",""), job.get("location",""), job.get("description",""))
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
    cur.execute("SELECT * FROM jobs WHERE rowid < ? ORDER BY rowid DESC LIMIT 1", (r["rowid"],))
    p = cur.fetchone()
    conn.close()
    if not p:
        raise HTTPException(status_code=404, detail="No previous jobs available")
    job = row_to_dict(p)
    ensure_url_field(job)

    if job.get("formatted_description"):
        job["description"] = job["formatted_description"]
        return job

    formatted = call_gpt_format(job.get("title",""), job.get("company",""), job.get("location",""), job.get("description",""))
    save_formatted_to_db(job.get("entity_id"), formatted)
    job["description"] = formatted
    job["formatted_description"] = formatted  # CRITICAL FIX: Update the field in the job object
    return job

# serve static files (unchanged)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
