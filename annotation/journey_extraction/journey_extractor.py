
import os
import sys
import argparse
import json    
import zipfile
import rarfile
import tarfile
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
from typing import List, Dict, Tuple, Set, Optional
from functools import lru_cache
from openai import OpenAI
from dotenv import load_dotenv
import backoff
from json import JSONDecodeError

# Load environment variables from .env if available"
load_dotenv()
              
# ----------------------------------------

# Prompt template for LLM journey extraction"

# ----------------------------------------
# Prompt template for LLM journey extraction
# ----------------------------------------
EXTRACTION_PROMPT = '''
You are a historical researcher. Extract all unique journeys/movements described in the following transcript snippet.
Return a JSON array where each element has these keys:
- from_location: str
- to_location: str
- approx_date: str or "Not mentioned"
- mode_of_transport: str or "Not mentioned"
- reason: str or "Not mentioned"
- context: short evidence sentence(s)

Transcript snippet:
"""
{chunk}
"""
'''

# ----------------------------------------
# File handling: archives and plaintext
# ----------------------------------------
def extract_texts_from_path(path: Path) -> List[Tuple[str, str]]:
    texts = []
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == '.txt':
            texts.append((path.name, path.read_text(encoding='utf-8', errors='ignore')))
        elif suffix in {'.zip', '.tar', '.gz', '.tgz', '.bz2', '.xz', '.rar'}:
            texts.extend(_extract_from_archive(path))
    elif path.is_dir():
        for child in path.rglob('*'):
            texts.extend(extract_texts_from_path(child))
    return texts

def _extract_from_archive(archive_path: Path) -> List[Tuple[str, str]]:
    texts = []
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as z:
            for name in z.namelist():
                if name.lower().endswith('.txt'):
                    texts.append((Path(name).name, z.read(name).decode('utf-8', 'ignore')))
    elif archive_path.suffix == '.rar':
        with rarfile.RarFile(archive_path, 'r') as r:
            for info in r.infolist():
                if info.filename.lower().endswith('.txt'):
                    texts.append((Path(info.filename).name, r.read(info).decode('utf-8', 'ignore')))
    else:
        with tarfile.open(archive_path, 'r') as t:
            for member in t.getmembers():
                if member.isfile() and member.name.lower().endswith('.txt'):
                    f = t.extractfile(member)
                    texts.append((Path(member.name).name, f.read().decode('utf-8', 'ignore')))
    return texts


# ----------------------------------------
# Chunking logic
# ----------------------------------------
def chunk_text(text: str, max_tokens: int = 10000) -> List[str]:
    paras = text.split('\n\n')
    chunks, current = [], []
    for p in paras:
        current.append(p)
        if sum(len(x.split()) for x in current) > max_tokens:
            chunks.append('\n\n'.join(current[:-1]))
            current = [p]
    if current:
        chunks.append('\n\n'.join(current))
    return chunks


# ----------------------------------------
# Client configuration
# ----------------------------------------
def get_llm_client(api_key: Optional[str], provider: str):
    if provider == 'openai':
        key = api_key or os.getenv("OPENAI_API_KEY")
    elif provider == 'groq':
        key = api_key or os.getenv("GROQ_API_KEY")
    else:
        print(f"[ERROR] Unsupported provider: {provider}", file=sys.stderr)
        sys.exit(1)

    if not key:
        print(
            f"[ERROR] API key for {provider} not provided.\n"
            f"Use --api-key or set {provider.upper()}_API_KEY in your environment/.env.",
            file=sys.stderr
        )
        sys.exit(1)

    return OpenAI(api_key=key, base_url=("https://api.groq.com/openai/v1" if provider == 'groq' else None))




# ----------------------------------------"
                },
                {

# LLM call with caching, rate limiting and error reporting"
                },
                {

# ----------------------------------------"
                },
                {

@lru_cache(maxsize=128)"
                },
                {

@backoff.on_exception(backoff.expo, Exception, max_tries=5)"
                },
                {

def call_llm_for_chunk_cached(chunk: str, model: str, api_key: Optional[str], provider: str) -> str:"
                },
                {

prompt = EXTRACTION_PROMPT.format(chunk=chunk)"
                },
                {

client = get_llm_client(api_key, provider)"
                },
                {

resp = client.chat.completions.create("
                },
                {

model=model,"
                },
                {

messages=[{'role': 'user', 'content': prompt}],"
                },
                {

temperature=0"
                },
                {

)"
                },
                {

return resp.choices[0].message.content"
                },
                {

"
                },
                {

def call_llm_for_chunk(chunk: str, model: str, api_key: Optional[str], provider: str, verbose: bool = False) -> List[Dict]:"
                },
                {

try:"
                },
                {

raw = call_llm_for_chunk_cached(chunk, model, api_key, provider)"
                },
                {

except Exception as e:"
                },
                {

if verbose:"
                },
                {

print(f\"[LLM ERROR] {e}\", file=sys.stderr)"
                },
                {

return []"
                },
                {

try:"
                },
                {

return json.loads(raw)"
                },
                {

except JSONDecodeError:"
                },
                {

if verbose:"
                },
                {

print(f\"[CHUNK ERROR] Invalid JSON for chunk starting '{chunk[:30]}...':\\n{raw}\\n---\", file=sys.stderr)"
                },
                {

return []"
                },
                {

"
                },
                {

# ----------------------------------------"
                },
                {

# Gazetteer & geocoding"
                },
                {

# ----------------------------------------"
                },
                {

def load_gazetteer(filepath: Path) -> Dict[str, Tuple[float, float]]:"
                },
                {

if not filepath.exists():"
                },
                {

return {}"
                },
                {

with open(filepath, 'r', encoding='utf-8') as f:"
                },
                {

return json.load(f)"
                },
                {

"
                },
                {

def save_gazetteer(gz: Dict[str, Tuple[float, float]], filepath: Path):"
                },
                {

with open(filepath, 'w', encoding='utf-8') as f:"
                },
                {

json.dump(gz, f, indent=2, ensure_ascii=False)"
                },
                {

"
                },
                {

@lru_cache(maxsize=256)"
                },
                {

@backoff.on_exception(backoff.expo, Exception, max_tries=5)"
                },
                {

def call_llm_geocode(loc: str, context: str, model: str, api_key: Optional[str], provider: str) -> str:"
                },
                {

prompt = f\"Estimate lat/lon for '{loc}' based on this context: {context}. Return as [lat, lon].\""
                },
                {

client = get_llm_client(api_key, provider)"
                },
                {

resp = client.chat.completions.create("
                },
                {

model=model,"
                },
                {

messages=[{'role': 'user', 'content': prompt}],"
                },
                {

temperature=0"
                },
                {

)"
                },
                {

return resp.choices[0].message.content"
                },
                {

"
                },
                {

def geocode_location("
                },
                {

loc: str,"
                },
                {

context: str,"
                },
                {

gaz: Dict[str, Tuple[float, float]],"
                },
                {

model: str,"
                },
                {

api_key: Optional[str],"
                },
                {

provider: str,"
                },
                {

verbose: bool = False"
                },
                {

) -> Tuple[float, float]:"
                },
                {

if loc in gaz:"
                },
                {

return gaz[loc]"
                },
                {

try:"
                },
                {

raw = call_llm_geocode(loc, context, model, api_key, provider)"
                },
                {

except Exception as e:"
                },
                {

if verbose:"
                },
                {

print(f\"[GEOCODE ERROR] {e}\", file=sys.stderr)"
                },
                {

return (0.0, 0.0)"
                },
                {

try:"
                },
                {

coords = json.loads(raw)"
                },
                {

except JSONDecodeError:"
                },
                {

if verbose:"
                },
                {

print(f\"[GEOCODE ERROR] Invalid JSON for '{loc}': {raw}\", file=sys.stderr)"
                },
                {

return (0.0, 0.0)"
                },
                {

gaz[loc] = tuple(coords)"
                },
                {

return gaz[loc]"
                },
                {

"
                },
                {

# ----------------------------------------"
                },
                {

# Worker for multiprocessing"
                },
                {

# ----------------------------------------"
                },
                {

def process_transcript("
                },
                {

item: Tuple[str, str],"
                },
                {

gazetteer: Path,"
                },
                {

output_path: Path,"
                },
                {

model: str,"
                },
                {

api_key: str,"
                },
                {

provider: str,"
                },
                {

verbose: bool = False"
                },
                {

) -> None:"
                },
                {

filename, txt = item"
                },
                {

if verbose:"
                },
                {

print(f\"[START] Processing {filename}\")"
                },
                {

try:"
                },
                {

chunks = chunk_text(txt)"
                },
                {

journeys = []"
                },
                {

gaz = load_gazetteer(gazetteer)"
                },
                {

for chunk in chunks:"
                },
                {

data = call_llm_for_chunk(chunk, model, api_key, provider, verbose=verbose)"
                },
                {

for entry in data:"
                },
                {

entry['transcript'] = filename"
                },
                {

entry['source_lat'], entry['source_lon'] = geocode_location("
                },
                {

entry['from_location'], entry['context'], gaz, model, api_key, provider, verbose=verbose"
                },
                {

)"
                },
                {

entry['target_lat'], entry['target_lon'] = geocode_location("
                },
                {

entry['to_location'], entry['context'], gaz, model, api_key, provider, verbose=verbose"
                },
                {

)"
                },
                {

journeys.append(entry)"
                },
                {

# dedupe"
                },
                {

seen = set()"
                },
                {

deduped = []"
                },
                {

for j in journeys:"
                },
                {

key = (j['from_location'], j['to_location'], j['approx_date'])"
                },
                {

if key not in seen:"
                },
                {

deduped.append(j)"
                },
                {

seen.add(key)"
                },
                {

# write results"
                },
                {

output_path.parent.mkdir(parents=True, exist_ok=True)"
                },
                {

with output_path.open('a', encoding='utf-8') as fout:"
                },
                {

for entry in deduped:"
                },
                {

fout.write(json.dumps(entry, ensure_ascii=False) + '\\n')"
                },
                {

save_gazetteer(gaz, gazetteer)"
                },
                {

if verbose:"
                },
                {

print(f\"[DONE] {filename}: {len(deduped)} journeys\")"
                },
                {

except Exception as e:"
                },
                {

print(f\"[ERROR] {filename} failed: {e}\", file=sys.stderr)"
                },
                {

"
                },
                {

# ----------------------------------------"
                },
                {

# Main"
                },
                {

# ----------------------------------------"
                },
                {

def main():"
                },
                {

parser = argparse.ArgumentParser("
                },
                {

description=\"Extract survivor journeys from transcripts and geocode locations\""
                },
                {

)"
                },
                {

parser.add_argument('-i', '--input', type=Path, required=True, help=\".txt file, folder, or archive to process\")"
                },
                {

parser.add_argument('-o', '--output', type=Path, default=Path('journey/journeys.jsonl'), help=\"Output JSONL file\")"
                },
                {

parser.add_argument('-g', '--gazetteer', type=Path, default=Path('journey/gazetteer.json'), help=\"Local gazetteer JSON file\")"
                },
                {

parser.add_argument('-w', '--workers', type=int, default=4, help=\"Number of parallel workers\")"
                },
                {

parser.add_argument('-k', '--api-key', type=str, default=None, help=\"LLM API key (overrides environment variable)\")"
                },
                {

parser.add_argument('-m', '--model', type=str, default='gpt-4', help=\"Model to use (e.g., gpt-4, llama3-70b-8192\")"
                },
                {

parser.add_argument('-p', '--provider', type=str, default='openai', choices=['openai', 'groq'], help=\"LLM provider\")"
                },
                {

parser.add_argument('-v', '--verbose', action='store_true', help=\"Verbose output\")"
                },
                {

args = parser.parse_args()"
                },
                {

"
                },
                {

items = extract_texts_from_path(args.input)"
                },
                {

if args.verbose:"
                },
                {

print(f\"[INFO] Found {len(items)} transcripts to process.\")"
                },
                {

"
                },
                {

if args.output.exists():"
                },
                {

args.output.unlink()"
                },
                {

"
                },
                {

with ProcessPoolExecutor(max_workers=args.workers) as exe:"
                },
                {

futures = ["
                },
                {

exe.submit("
                },
                {

process_transcript,"
                },
                {

item,"
                },
                {

args.gazetteer,"
                },
                {

args.output,"
                },
                {

args.model,"
                },
                {

args.api_key,"
                },
                {

args.provider,"
                },
                {

args.verbose"
                },
                {

) for item in items"
                },
                {

]"
                },
                {

for fut in tqdm(as_completed(futures), total=len(futures), desc=\"Transcripts\"):"
                },
                {

try:"
                },
                {

fut.result()"
                },
                {

except Exception as e:"
                },
                {

print(f\"[ERROR] A transcript failed: {e}\", file=sys.stderr)"
                },
                {

"
                },
                {

if args.verbose:"
                },
                {

print(f\"Extraction complete. Results saved to {args.output}\")"
                },
                {

print(f\"Updated gazetteer saved to {args.gazetteer}\")"
                },
                {

"
                },
                {

if __name__ == '__main__':"
                },
                {

main()"
                }
            ],
            "token_count": 1919
        }
    ]
}