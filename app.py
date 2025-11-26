#!/usr/bin/env python3
"""
Production-ready TDS LLM Quiz Solver - app.py

Features:
- POST /task endpoint (email, secret, url)
- Secret validation (HTTP 400/403 semantics)
- Playwright to render JS pages and extract data
- Deterministic handlers: HTML tables, CSV/XLSX, PDFs (pdfplumber)
- Audio detection + transcription via Whisper API
- LLM fallback stub (configurable via env)
- Timeouts & size checks (downloads and final JSON < 1MB)
- Useful metadata returned for debugging/evaluation
"""

import os
import io
import re
import time
import json
import base64
import mimetypes
import asyncio
import logging
from typing import Optional, Any, Dict, Tuple

from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
import httpx
import pandas as pd
import pdfplumber
from playwright.async_api import async_playwright


# -------------------------
# Configuration via ENV
# -------------------------
EXPECTED_SECRET = os.getenv("QUIZ_SECRET")

LLM_API_URL = os.getenv("LLM_API_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")

WHISPER_API_URL = os.getenv("WHISPER_API_URL")
WHISPER_API_KEY = os.getenv("WHISPER_API_KEY")


MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(10 * 1024 * 1024)))
TOTAL_TIMEOUT = int(os.getenv("TOTAL_TIMEOUT", "170"))
FINAL_JSON_LIMIT = int(os.getenv("FINAL_JSON_LIMIT", str(1 * 1024 * 1024)))
PLAYWRIGHT_PAGE_TIMEOUT = int(os.getenv("PLAYWRIGHT_PAGE_TIMEOUT", "60000"))
DOWNLOAD_TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT", "30"))

# -------------------------
# App & logging
# -------------------------
app = Flask(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("tds-llm")

# -------------------------
# Pydantic payload model
# -------------------------
class TaskPayload(BaseModel):
    email: str
    secret: str
    url: str


# -------------------------
# Helpers - JSON error
# -------------------------
def json_error(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


# -------------------------
# Entrypoints
# -------------------------
@app.route("/", methods=["GET"])
def index():
    return "TDS LLM Quiz Solver - healthy"


@app.route("/task", methods=["POST"])
def task_receiver():
    """
    Validate JSON & secret. Run async solver with TOTAL_TIMEOUT.
    Respond HTTP 400 for invalid JSON/payload, 403 for invalid secret.
    On valid secret, always return HTTP 200 with solver result body.
    """
    start_ts = time.time()
    try:
        body = request.get_json(force=True)
    except Exception:
        return json_error("invalid json", 400)

    try:
        payload = TaskPayload(**body)
    except ValidationError as e:
        return json_error("invalid payload: " + str(e), 400)

    if payload.secret != EXPECTED_SECRET:
        return json_error("invalid secret", 403)

    # Run solver with asyncio timeout guard
    try:
        result = asyncio.run(asyncio.wait_for(
            solve_quiz_and_submit(payload, start_ts),
            timeout=TOTAL_TIMEOUT
        ))
    except asyncio.TimeoutError:
        logger.warning("Solver exceeded total timeout of %s seconds", TOTAL_TIMEOUT)
        return jsonify({"status": "timeout", "message": f"Solver exceeded {TOTAL_TIMEOUT}s limit"}), 200
    except Exception as e:
        logger.exception("Unexpected error while running solver")
        return jsonify({"status": "error", "error": str(e)}), 200

    elapsed = time.time() - start_ts
    if isinstance(result, dict):
        result["elapsed_s"] = elapsed
    return jsonify(result), 200


# -------------------------
# Core solver
# -------------------------
async def solve_quiz_and_submit(payload: TaskPayload, start_ts: float) -> Dict[str, Any]:
    logger.info("Received task for URL: %s", payload.url)
    meta: Dict[str, Any] = {"log": []}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            # Navigate to URL
            meta["log"].append("navigating")
            await page.goto(payload.url, timeout=PLAYWRIGHT_PAGE_TIMEOUT)

            try:
                await page.wait_for_load_state("networkidle", timeout=20000)
            except Exception:
                meta["log"].append("networkidle-failed")

            await asyncio.sleep(0.5)

            page_html = await page.content()
            try:
                page_text = await page.inner_text("body", timeout=5000)
            except Exception:
                page_text = re.sub(r"<[^>]*>", " ", page_html)

            # 1) find submit URL
            submit_url = await extract_submit_url(page_html, page_text, page)
            meta["submit_url"] = submit_url

            # 2) attempt heuristics
            answer, heur_meta = await extract_answer_heuristics(page, page_text, page, payload)
            meta.update(heur_meta)

            # 3) LLM fallback
            if answer is None:
                meta["log"].append("heuristics_failed")
                if LLM_API_URL:
                    answer = await llm_extract_answer_stub(page_text, payload.url)
                    meta["via"] = "llm"
                else:
                    await browser.close()
                    return {
                        "status": "unable_to_extract",
                        "reason": "no_llm_configured",
                        "meta": meta,
                        "page_snippet": page_html[:2000]
                    }

            # 4) Build final answer JSON
            submission = {
                "email": payload.email,
                "secret": payload.secret,
                "url": payload.url,
                "answer": answer
            }

            json_bytes = json.dumps(submission).encode("utf-8")
            if len(json_bytes) > FINAL_JSON_LIMIT:
                await browser.close()
                return {"status": "answer_too_large", "size_bytes": len(json_bytes), "meta": meta}

            if not submit_url:
                await browser.close()
                return {"status": "no_submit_url_found", "meta": meta, "page_snippet": page_html[:2000]}

            meta["log"].append("submitting_answer")
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(submit_url, json=submission)
                try:
                    submit_resp = r.json()
                except Exception:
                    submit_resp = {"status_code": r.status_code, "text": r.text}

            await browser.close()
            return {"status": "submitted", "submit_response": submit_resp, "meta": meta}

        except Exception as e:
            await browser.close()
            logger.exception("Solve error")
            meta["error"] = str(e)
            return {"status": "error", "error": str(e), "meta": meta}


# -------------------------
# Extraction helpers
# -------------------------
async def extract_submit_url(page_html: str, page_text: str, page_handle) -> Optional[str]:
    # 1) form action
    try:
        action = await page_handle.eval_on_selector("form", "el => el.action", strict=False)
        if action:
            return action
    except Exception:
        pass

    # 2) contains "/submit"
    m = re.search(r"https?://[^\s'\"<>]*?/submit[^\s'\"<>]*", page_html)
    if m:
        return m.group(0)

    # 3) from <pre> JSON
    try:
        pre = await page_handle.eval_on_selector("pre", "el => el.textContent", strict=False)
        if pre:
            try:
                js = json.loads(pre)
                if "submit" in js:
                    return js["submit"]
            except Exception:
                pass
    except Exception:
        pass

    # 4) fallback
    m2 = re.search(r"https?://[^\s'\"<>]*?(submit|answer|response)[^\s'\"<>]*", page_text, flags=re.I)
    if m2:
        return m2.group(0)
    return None


async def extract_answer_heuristics(page, page_text: str, page_handle, payload: TaskPayload):
    meta = {"steps": []}

    # 1) <pre> JSON block
    try:
        pre_json = await page_handle.eval_on_selector("pre", "el => el.textContent", strict=False)
    except Exception:
        pre_json = None
    if pre_json:
        meta["steps"].append("pre_json")
        try:
            doc = json.loads(pre_json)
            if isinstance(doc, dict) and "answer" in doc:
                return doc["answer"], meta
        except Exception:
            pass

    # 2) HTML table "value" sum
    try:
        table_sum = await page_handle.eval_on_selector_all("table", """
        (tables) => {
            function n(s){return (s||'').toString().trim().toLowerCase();}
            for(const t of tables){
                const headers = Array.from(t.querySelectorAll('th')).map(x=>n(x.innerText));
                const idx=headers.indexOf('value');
                if(idx>=0){
                    let sum=0;
                    for(const r of Array.from(t.querySelectorAll('tbody tr'))){
                        const cols=Array.from(r.querySelectorAll('td'));
                        if(cols.length<=idx) continue;
                        const txt=cols[idx].innerText.replace(/[^0-9+\\-.,]/g,'').replace(',','');
                        const v=parseFloat(txt);
                        if(!isNaN(v)) sum+=v;
                    }
                    return sum;
                }
            }
            return null;
        }
        """)
        if table_sum:
            meta["steps"].append("html_table_value_sum")
            return table_sum, meta
    except Exception:
        pass

    # 3) PDF download
    pdf_url = None
    m_pdf = re.search(r"https?://[^\s'\"<>]+\\.pdf", page_text, flags=re.I)
    if m_pdf:
        pdf_url = m_pdf.group(0)
        meta["steps"].append("found_pdf_url")
        pdf_bytes = await download_file(pdf_url)
        if pdf_bytes:
            val = parse_pdf_sum_value_on_page(pdf_bytes, page_number=2, column_name="value")
            if val is not None:
                meta["steps"].append("pdf_parsed")
                return val, meta

    # 4) CSV/XLSX
    m_csv = re.search(r"https?://[^\s'\"<>]+\\.(csv|xlsx|xls)", page_text, flags=re.I)
    if m_csv:
        csv_url = m_csv.group(0)
        meta["steps"].append("found_csv_xlsx")
        fbytes = await download_file(csv_url)
        if fbytes:
            df = None
            try:
                df = pd.read_csv(io.BytesIO(fbytes))
            except Exception:
                try:
                    df = pd.read_excel(io.BytesIO(fbytes))
                except Exception:
                    df = None
            if df is not None:
                for col in df.columns:
                    if str(col).strip().lower() == "value":
                        s = df[col].apply(pd.to_numeric, errors="coerce").sum(skipna=True)
                        meta["steps"].append("csv_value_sum")
                        return float(s), meta

    # 5) Audio
    audio_match = re.search(r"https?://[^\s'\"<>]+\\.(wav|mp3|ogg|m4a|flac|aac)", page_text, flags=re.I)
    if audio_match:
        audio_url = audio_match.group(0)
        meta["steps"].append("found_audio")
        audio_bytes = await download_file(audio_url)
        if audio_bytes:
            transcript = await transcribe_audio(audio_bytes, filename=audio_url.split("/")[-1])
            if transcript:
                meta["steps"].append("audio_transcribed")
                return transcript, meta

    # 6) Base64 atob
    b64_match = re.search(r"atob\\(['\"]([A-Za-z0-9+/=\\n\\r]+)['\"]\\)", page_text)
    if b64_match:
        try:
            decoded = base64.b64decode(b64_match.group(1))
            m = re.search(rb"{.*}", decoded, flags=re.S)
            if m:
                try:
                    js = json.loads(m.group(0).decode("utf-8"))
                    if "answer" in js:
                        meta["steps"].append("base64_json_pre")
                        return js["answer"], meta
                except Exception:
                    pass
        except Exception:
            pass

    return None, meta


# -------------------------
# download file
# -------------------------
async def download_file(url: str) -> Optional[bytes]:
    logger.info("Downloading: %s", url)
    try:
        async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT) as client:
            r = await client.get(url, follow_redirects=True)
            if r.status_code != 200:
                return None

            content_length = int(r.headers.get("content-length") or 0)
            if content_length > MAX_FILE_BYTES:
                return None

            data = r.content
            if len(data) > MAX_FILE_BYTES:
                return None

            return data
    except Exception:
        logger.exception("download_file exception")
        return None


# -------------------------
# PDF parsing
# -------------------------
def parse_pdf_sum_value_on_page(pdf_bytes: bytes, page_number: int, column_name: str):
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if page_number - 1 >= len(pdf.pages):
                return None
            page = pdf.pages[page_number - 1]
            tables = page.extract_tables()
            if not tables:
                return None
            for table in tables:
                headers = [str(x).strip().lower() for x in table[0]]
                if column_name.lower() in headers:
                    idx = headers.index(column_name.lower())
                    s = 0
                    for row in table[1:]:
                        try:
                            s += float(str(row[idx]).replace(",", ""))
                        except:
                            pass
                    return s
    except:
        return None
    return None


# -------------------------
# Whisper transcription
# -------------------------
async def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav"):
    if not WHISPER_API_URL or not WHISPER_API_KEY:
        logger.warning("Whisper not configured")
        return None

    mime = mimetypes.guess_type(filename)[0] or "audio/wav"
    files = {
        "file": (filename, audio_bytes, mime),
        "model": (None, "whisper-1"),
    }
    headers = {"Authorization": f"Bearer {WHISPER_API_KEY}"}

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(WHISPER_API_URL, files=files, headers=headers)
            data = r.json()
            if "text" in data:
                return data["text"].strip()
            return None
    except:
        return None


# -------------------------
# LLM fallback
# -------------------------
async def llm_extract_answer_stub(page_text: str, url: str):
    if not LLM_API_URL:
        return None

    headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
    payload = {
        "prompt": f"Extract the answer from this quiz page:\nURL: {url}\n\n{page_text[:6000]}",
        "max_tokens": 256
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(LLM_API_URL, json=payload, headers=headers)
        try:
            data = r.json()
            if "answer" in data:
                return data["answer"]
            if "text" in data:
                return data["text"]
            return r.text.strip()
        except:
            return None


# -------------------------
# run
# -------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
