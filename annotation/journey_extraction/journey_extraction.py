import os
import json
import argparse
import openai
import requests
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import Tuple, Dict, List

# this reads the .env file and adds its vars to os.environ
from dotenv import load_dotenv
load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")

# -- Configuration -------------------------------------------------------------
PROMPT_TEMPLATE = """
You are an expert analyst of historical survivor testimonies. Your task is to extract all the journeys embarked on by the narrator from the transcript below.

Instructions:
1. Identify all instances where the narrator travels from one location to another.
2. For each journey, extract the following:
   - `from_location`: The place departed from.
   - `to_location`: The destination.
   - `approx_date`: An approximate date or time period if mentioned.
   - `mode_of_transport`: If mentioned (e.g., train, walking, cart).
   - `reason`: Brief reason or context for the journey.
3. Arrange all journeys in **chronological order** based on available clues.

Return the result as a **JSON array** of objects, each with the keys: `from_location`, `to_location`, `approx_date`, `mode_of_transport`, and `reason`.

Transcript:
"{text}"
"""

# -- Helper Functions ----------------------------------------------------------
def get_api_key(cli_key: str = None) -> str:
    return cli_key or os.getenv("OPENAI_API_KEY")

def call_gpt(prompt: str, model: str) -> str:
    messages = [
        {"role": "system", "content": "You are a careful and structured extractor of data from historical testimonies."},
        {"role": "user", "content": prompt}
    ]

    if model.startswith("o"):  # for o-X models like o3, o3-mini, o4
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages
        )
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.0
        )

    return response.choices[0].message.content


def geocode_location(location: str) -> Tuple[float, float]:
    """
    Geocode a location name to (lat, lon). Returns (None, None) on failure.
    """
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": location, "format": "json", "limit": 1}
        headers = {"User-Agent": "LLM-Journey-Extractor"}
        resp = requests.get(url, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if data:
            lat, lon = float(data[0]['lat']), float(data[0]['lon'])
            return lat, lon
    except:
        pass
    return None, None


def load_transcripts(input_path: str) -> Dict[str, str]:
    transcripts = {}
    if os.path.isdir(input_path):
        for fn in os.listdir(input_path):
            if fn.lower().endswith('.txt'):
                with open(os.path.join(input_path, fn), 'r', encoding='utf-8') as f:
                    transcripts[fn] = f.read()
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            transcripts[os.path.basename(input_path)] = f.read()
    return transcripts


def validate_journey(j: dict) -> bool:
    keys = {"from_location", "to_location", "approx_date", "mode_of_transport", "reason"}
    return keys.issubset(j.keys())

# -- Core Pipeline -------------------------------------------------------------
class JourneyExtractor:
    def __init__(self, model: str, api_key: str, output_dir: str, workers: int = None):
        openai.api_key = api_key
        self.model = model
        self.out = output_dir
        os.makedirs(self.out, exist_ok=True)
        self.workers = workers or max(1, cpu_count() - 1)

    def extract(self, transcripts: Dict[str, str]) -> List[dict]:
        results = []
        for name, text in tqdm(transcripts.items(), desc="LLM Extraction"):
            prompt = PROMPT_TEMPLATE.format(text=text)

            # 1) Call the API and let exceptions bubble up so you see the full traceback
            resp = call_gpt(prompt, self.model)
            # print(f"\n--- DEBUG: Raw response for {name} ---\n{resp!r}\n") ## uncommend for raw responses

            # 2) Strip markdown fences if present
            stripped = resp.strip()
            if stripped.startswith("```"):
                parts = stripped.split("```")
                if len(parts) >= 3:
                    resp = parts[1].strip()
                    # print(f"--- DEBUG: Stripped code block for {name} ---\n{resp!r}\n") ## uncommend for raw responses
                else:
                    print(f"[Warning][{name}] Malformed fences; skipping strip.")

            # 2.5) Remove leading 'json\n' if present
            if resp.lower().startswith("json\n"):
                resp = resp[5:].lstrip()
                print(f"--- DEBUG: Removed 'json\\n' prefix for {name} ---\n{resp!r}\n") ## uncommend for raw responses

            # 3) Try JSON parsing, now catching only JSONDecodeError
            try:
                journeys = json.loads(resp)
            except json.JSONDecodeError as e:
                print(f"[Error][JSONDecodeError][{name}]: {e}")
                pos = e.pos
                snippet = resp[max(0, pos-20):pos+20]
                print(f"…{snippet!r}…")
                continue

            # 4) Validate each journey
            for j in journeys:
                if validate_journey(j):
                    j['transcript'] = name
                    results.append(j)
                else:
                    print(f"[Warning][Validation Failed][{name}]: {j!r}")

        return results

    def geocode_all(self, journeys: List[dict]) -> List[dict]:
        # gather unique locations
        locs = set()
        for j in journeys:
            locs.update([j['from_location'], j['to_location']])
        locs = list(locs)

        # parallel geocoding
        with Pool(self.workers) as pool:
            coords = pool.map(geocode_location, locs)
        geo_map = dict(zip(locs, coords))

        # attach to journeys
        for j in journeys:
            j['source_lat'], j['source_lon'] = geo_map.get(j['from_location'], (None, None))
            j['target_lat'], j['target_lon'] = geo_map.get(j['to_location'], (None, None))
        return journeys

    def export_jsonl(self, journeys: List[dict]):
        path = os.path.join(self.out, f'journeys_{self.model}.jsonl')
        with open(path, 'w', encoding='utf-8') as f:
            for j in journeys:
                f.write(json.dumps(j) + "\n")
        print(f"JSONL saved: {path}")

    def export_csv(self, journeys: List[dict]):
        df = pd.DataFrame(journeys)
        path = os.path.join(self.out, f'journeys_{self.model}.csv')
        df.to_csv(path, index=False)
        print(f"CSV saved: {path}")

    def run_all(self, transcripts: Dict[str, str]):
        journeys = self.extract(transcripts)
        journeys = self.geocode_all(journeys)
        self.export_jsonl(journeys)
        self.export_csv(journeys)

# -- CLI ----
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract, geocode, and export journeys.")
    parser.add_argument('--input', '-i', required=True, help="Path to .txt file or folder.")
    parser.add_argument('--output', '-o', required=True, default="journeys", help="Output directory.")
    parser.add_argument('--model', '-M', default="gpt-4-turbo", help="OpenAI model to use.")
    parser.add_argument('--api_key', help="OpenAI API key (or set via environment).")
    args = parser.parse_args()
    api_key = get_api_key(args.api_key)
    trans = load_transcripts(args.input)
    ext = JourneyExtractor(model=args.model, api_key=api_key, output_dir=args.output)
    ext.run_all(trans)