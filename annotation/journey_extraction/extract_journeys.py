import os
import sys
import json
import ast
import argparse
import asyncio
from typing import Dict, List

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from dotenv import load_dotenv
from tqdm import tqdm

# —— 0) Load environment variables — including OPENAI_API_KEY — from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("[Error] OPENAI_API_KEY not set; please add it to your .env", file=sys.stderr)
    sys.exit(1)

# —— 1) Supported models & per-model prompt templates
SUPPORTED_MODELS = ["gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo-1106", "gpt-4.5-preview"]
PROMPT_TEMPLATES = {
    "default": (
        "Extract a chronological list of journeys from the following transcript.  "
        "Output **only** valid JSON array of objects with keys: "
        "`from_location`, `to_location`, `approx_date`, `mode_of_transport`, `reason`.\n\n"
        "Transcript:\n\n{text}\n\n"
    ),
    "gpt-3.5-turbo": (
        "You are a JSON machine.  Given the transcript below, produce a pure JSON array of "
        "journey objects, each with exactly these fields: "
        "`from_location`, `to_location`, `approx_date`, `mode_of_transport`, `reason`.\n\n"
        "Transcript:\n\n{text}\n\n"
    ),
}

def get_prompt(model: str, text: str) -> str:
    key = "gpt-3.5-turbo" if model.startswith("gpt-3.5") else "default"
    return PROMPT_TEMPLATES[key].format(text=text)

# —— 2) Async GPT call with automatic retry
@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(min=1, max=60),
    retry=retry_if_exception_type((openai.error.RateLimitError, openai.error.Timeout)),
)
async def call_gpt_async(prompt: str, model: str) -> str:
    resp = await openai.ChatCompletion.acreate(
        model=model,
        messages=[{"role": "system", "content": "You extract journeys."},
                  {"role": "user",   "content": prompt}],
        temperature=0.0,
    )
    return resp.choices[0].message.content

# —— 3) Per-transcript processing
async def process_transcript(name: str, text: str, model: str) -> List[dict]:
    if not text.strip():
        return []

    raw = await call_gpt_async(get_prompt(model, text), model)
    s = raw.strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1].strip()
    if s.lower().startswith("json\n"):
        s = s[5:].lstrip()
    try:
        journeys = json.loads(s)
    except json.JSONDecodeError:
        try:
            journeys = ast.literal_eval(s)
        except Exception:
            print(f"[Error][{name}] Could not parse JSON: {s[:100]!r}")
            return []

    valid = []
    for j in journeys:
        if all(k in j for k in ("from_location","to_location","approx_date","mode_of_transport","reason")):
            j["transcript"] = name
            valid.append(j)
        else:
            print(f"[Warning][{name}] Invalid entry skipped: {j!r}")
    return valid

# —— 4) Batch-oriented async extractor
async def extract_async(transcripts: Dict[str,str], model: str) -> List[dict]:
    tasks = [
        process_transcript(name, text, model)
        for name, text in transcripts.items()
    ]
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Extracting"):
        res = await coro
        results.extend(res)
    return results

# —— 5) Sync wrapper
def extract(transcripts: Dict[str,str], model: str) -> List[dict]:
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Model '{model}' not supported. Choose from {SUPPORTED_MODELS}")
    return asyncio.run(extract_async(transcripts, model))

# —— 6) CLI entrypoint
def cli_main():
    parser = argparse.ArgumentParser(description="Journey extraction CLI")
    parser.add_argument("-i","--input", required=True, help="Folder or file of transcripts (JSON or txt)")
    parser.add_argument("-o","--output", required=True, help="Output JSONL path")
    parser.add_argument("-m","--model", default="gpt-4-turbo", help=f"One of {SUPPORTED_MODELS}")
    args = parser.parse_args()

    transcripts = {}
    if os.path.isdir(args.input):
        for fn in os.listdir(args.input):
            path = os.path.join(args.input, fn)
            with open(path, encoding="utf-8") as f:
                transcripts[fn] = f.read()
    else:
        with open(args.input, encoding="utf-8") as f:
            transcripts[os.path.basename(args.input)] = f.read()

    extracted = extract(transcripts, args.model)

    with open(args.output, "w", encoding="utf-8") as out:
        for j in extracted:
            out.write(json.dumps(j, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(extracted)} journeys to {args.output}")

# —— 7) CLI-only runner
if __name__ == "__main__":
    cli_main()
