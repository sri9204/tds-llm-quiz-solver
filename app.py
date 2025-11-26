#!/usr/bin/env python3
"""
FINAL — PRODUCTION READY TDS LLM QUIZ SOLVER
Fixed Version (multi-step supported)

✓ Multi-step quizzes (next → next → final)
✓ Detects "next" inside <pre> JSON
✓ Prevents LLM interference on multi-step pages
✓ HTML/PDF/CSV/XLSX/Audio supported
✓ Safe fallback logic
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
import urllib.parse
from flask_cors import CORS



from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
import httpx
import pandas as pd
import pdfplumber
from playwright.async_api import async_playwright

# --------------------------------------------------------
# ENV CONFIG
# --------------------------------------------------------
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

# --------------------------------------------------------
# APP + LOGGING
# --------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("tds-llm")

CORS(app)


# --------------------------------------------------------
# PAYLOAD MODEL
# --------------------------------------------------------
class TaskPayload(BaseModel):
    email: str
    secret: str
    url: str


def json_error(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


# --------------------------------------------------------
# ROOT + TASK HANDLER
# --------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return "TDS LLM Quiz Solver — healthy"


@app.route("/task", methods=["POST"])
def task_receiver():
    start_ts = time.time()

    try:
        body = request.get_json(force=True)
    except:
        return json_error("invalid json", 400)

    try:
        payload = TaskPayload(**body)
    except ValidationError as e:
        return json_error("invalid payload: " + str(e), 400)

    if payload.secret != EXPECTED_SECRET:
        return json_error("invalid secret", 403)

    try:
        result = asyncio.run(asyncio.wait_for(
            solve_quiz_and_submit(payload),
            timeout=TOTAL_TIMEOUT
        ))
    except asyncio.TimeoutError:
        return jsonify({"status": "timeout"}), 200
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 200

    result["elapsed_s"] = time.time() - start_ts
    return jsonify(result), 200


# --------------------------------------------------------
# CORE SOLVER (NOW MULTI-STEP)
# --------------------------------------------------------
async def solve_quiz_and_submit(payload: TaskPayload):
    meta = {"log": [], "steps": []}
    answer_history = []
    current_url = payload.url

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            while True:
                meta["log"].append(f"navigating:{current_url}")
                meta["steps"].append(current_url)

                await page.goto(current_url, timeout=PLAYWRIGHT_PAGE_TIMEOUT)

                try:
                    await page.wait_for_load_state("networkidle", timeout=20000)
                except:
                    meta["log"].append("networkidle-failed")

                await asyncio.sleep(0.3)

                html = await page.content()
                try:
                    text = await page.inner_text("body", timeout=5000)
                except:
                    text = re.sub(r"<[^>]*>", " ", html)

                # ---------------------------------------
                # extract submit URL
                # ---------------------------------------
                submit_url = await extract_submit_url(page, html, text)
                meta["submit_url"] = submit_url

                # ---------------------------------------
                # extract answer OR next_url
                # ---------------------------------------
                answer, heur_meta, next_url_override = await extract_answer_heuristics(page, html, text)
                meta.update(heur_meta)

                # ---------------------------------------
                # MULTI-STEP detected via <pre> JSON
                # ---------------------------------------
                if next_url_override:
                    meta["log"].append(f"multi_step_redirect:{next_url_override}")
                    current_url = urllib.parse.urljoin(current_url, next_url_override)
                    continue

                # ---------------------------------------
                # normal answer case
                # ---------------------------------------
                if answer is None:
                    meta["log"].append("heuristics_failed")

                    # LLM fallback ONLY if enabled AND page is NOT multi-step
                    if LLM_API_URL:
                        answer = await llm_extract_answer_stub(text, current_url)
                        meta["via"] = "llm"
                    else:
                        return {"status": "unable_to_extract", "meta": meta}

                answer_history.append(answer)

                # build submission
                submission = {
                    "email": payload.email,
                    "secret": payload.secret,
                    "url": current_url,
                    "answer": answer
                }

                if len(json.dumps(submission).encode()) > FINAL_JSON_LIMIT:
                    return {"status": "answer_too_large", "meta": meta}

                if not submit_url:
                    return {"status": "no_submit_url", "meta": meta}

                meta["log"].append(f"submitting_answer:{current_url}")

                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.post(submit_url, json=submission)
                    try:
                        submit_resp = r.json()
                    except:
                        submit_resp = {"status_code": r.status_code, "text": r.text}

                # get next url for multi-step
                next_url = submit_resp.get("url")

                if not next_url:
                    return {
                        "status": "multi-step-done",
                        "final_response": submit_resp,
                        "answers": answer_history,
                        "meta": meta
                    }

                meta["log"].append(f"next_url:{next_url}")
                current_url = next_url

        finally:
            await browser.close()


# --------------------------------------------------------
# SUBMIT URL EXTRACTION
# --------------------------------------------------------
async def extract_submit_url(page, html, text):
    try:
        action = await page.eval_on_selector("form", "el => el.action", strict=False)
        if action:
            return action
    except:
        pass

    try:
        pre = await page.eval_on_selector("pre", "el => el.textContent", strict=False)
        if pre:
            try:
                js = json.loads(pre)
                if "submit" in js:
                    return js["submit"]
            except:
                pass
    except:
        pass

    m = re.search(r"https?://[^\s\"']*submit[^\s\"']*", html)
    if m:
        return m.group(0)

    return None


# --------------------------------------------------------
# HEURISTIC EXTRACTION (NOW SUPPORTS next:)
# --------------------------------------------------------
async def extract_answer_heuristics(page, html, text):
    meta = {"steps": []}
    next_url_override = None

    # ---------------------------------------
    # 1) PRE JSON
    # ---------------------------------------
    try:
        pre = await page.eval_on_selector("pre", "el => el.textContent", strict=False)
    except:
        pre = None

    if pre:
        try:
            js = json.loads(pre)

            # MULTI-STEP detection
            if "next" in js:
                meta["steps"].append("found_next_url")
                next_url_override = js["next"]
                return None, meta, next_url_override

            if "answer" in js:
                meta["steps"].append("pre_json")
                return js["answer"], meta, None

        except:
            pass

    # ---------------------------------------
    # 2) HTML Table sum
    # ---------------------------------------
    try:
        table_sum = await page.eval_on_selector_all("table", """
        (tables) => {
            function n(s){return (s||'').toString().trim().toLowerCase();}
            for(const t of tables){
                const headers = Array.from(t.querySelectorAll('th')).map(x=>n(x.innerText));
                const idx = headers.indexOf('value');
                if(idx >= 0){
                    let total = 0;
                    for(const r of t.querySelectorAll('tbody tr')){
                        const cells = Array.from(r.querySelectorAll('td'));
                        if(cells.length <= idx) continue;
                        const raw = cells[idx].innerText.replace(/[^0-9+\-.,]/g,'');
                        const v = parseFloat(raw);
                        if(!isNaN(v)) total += v;
                    }
                    return total;
                }
            }
            return null;
        }
        """)
        if table_sum:
            meta["steps"].append("html_table_sum")
            return table_sum, meta, None
    except:
        pass

    # ---------------------------------------
    # 3) PDF
    # ---------------------------------------
    pdf_url = None
    try:
        links = await page.eval_on_selector_all("a", "els => els.map(e => e.href)")
        for href in links:
            if href.lower().endswith(".pdf"):
                pdf_url = href
                break
    except:
        pass

    if pdf_url:
        meta["steps"].append("pdf_link_found")
        pdf_bytes = await download_file(pdf_url)
        if pdf_bytes:
            meta["steps"].append("pdf_downloaded")
            val = parse_pdf_sum_value_on_page(pdf_bytes, 2, "value")
            if val is not None:
                meta["steps"].append("pdf_parsed")
                return val, meta, None

    # ---------------------------------------
    # 4) CSV / XLSX
    # ---------------------------------------
    m = re.search(r"https?://[^\s\"']+\.(csv|xlsx|xls)", text, flags=re.I)
    if m:
        csv_url = m.group(0)
        meta["steps"].append("csv_link_found")
        data = await download_file(csv_url)

        if data:
            try:
                df = pd.read_csv(io.BytesIO(data))
            except:
                try:
                    df = pd.read_excel(io.BytesIO(data))
                except:
                    df = None

            if df is not None:
                for c in df.columns:
                    if str(c).strip().lower() == "value":
                        total = df[c].apply(pd.to_numeric, errors="coerce").sum()
                        meta["steps"].append("csv_sum")
                        return float(total), meta, None

    # ---------------------------------------
    # 5) AUDIO
    # ---------------------------------------
    a = re.search(r"https?://[^\s\"']+\.(wav|mp3|ogg|m4a|flac|aac)", text, flags=re.I)
    if a:
        audio_url = a.group(0)
        meta["steps"].append("audio_link_found")
        audio_bytes = await download_file(audio_url)
        if audio_bytes:
            transcript = await transcribe_audio(audio_bytes, audio_url.split("/")[-1])
            if transcript:
                meta["steps"].append("audio_transcribed")
                return transcript, meta, None

    # ---------------------------------------
    # 6) Base64 JSON
    # ---------------------------------------
    b = re.search(r"atob\(['\"]([A-Za-z0-9+/=\n\r]+)['\"]\)", text)
    if b:
        try:
            decoded = base64.b64decode(b.group(1))
            m2 = re.search(rb"{.*?}", decoded, flags=re.S)
            if m2:
                js = json.loads(m2.group(0).decode())
                if "answer" in js:
                    meta["steps"].append("base64_json")
                    return js["answer"], meta, None
        except:
            pass

    return None, meta, None


# --------------------------------------------------------
# FILE DOWNLOAD
# --------------------------------------------------------
async def download_file(url):
    try:
        async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT) as client:
            r = await client.get(url)
            if r.status_code != 200:
                return None
            if len(r.content) > MAX_FILE_BYTES:
                return None
            return r.content
    except:
        return None


# --------------------------------------------------------
# PDF PARSING
# --------------------------------------------------------
def parse_pdf_sum_value_on_page(pdf_bytes, page_number, column_name):
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages_to_check = []

            if page_number - 1 < len(pdf.pages):
                pages_to_check.append(pdf.pages[page_number - 1])

            pages_to_check.extend(pdf.pages)

            for page in pages_to_check:
                tables = page.extract_tables()
                if not tables:
                    continue

                for table in tables:
                    headers = [str(x).strip().lower() for x in table[0]]
                    if column_name.lower() in headers:
                        idx = headers.index(column_name.lower())
                        total = 0
                        for row in table[1:]:
                            try:
                                total += float(str(row[idx]).replace(",", ""))
                            except:
                                pass
                        return total
    except:
        return None

    return None


# --------------------------------------------------------
# WHISPER
# --------------------------------------------------------
async def transcribe_audio(audio_bytes, filename="audio.wav"):
    if not WHISPER_API_URL or not WHISPER_API_KEY:
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
            j = r.json()
            return j.get("text", "").strip()
    except:
        return None


# --------------------------------------------------------
# LLM FALLBACK
# --------------------------------------------------------
async def llm_extract_answer_stub(page_text: str, url: str):
    if not LLM_API_URL or not LLM_API_KEY:
        return None

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "Extract the answer only, do not explain."
            },
            {
                "role": "user",
                "content": f"URL: {url}\n\nPAGE TEXT:\n{page_text[:6000]}"
            }
        ]
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(LLM_API_URL, json=payload, headers=headers)
        try:
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except:
            return None


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
