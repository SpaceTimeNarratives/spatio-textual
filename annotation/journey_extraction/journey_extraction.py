import os
import json
import time
import argparse
import openai
import requests
import pandas as pd
import folium
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pyvis.network import Network
from typing import Tuple, Dict, List

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
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful and structured extractor of data from historical testimonies."},
            {"role": "user", "content": prompt}
        ],
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
            try:
                resp = call_gpt(prompt, self.model)
                journeys = json.loads(resp)
                for j in journeys:
                    if validate_journey(j):
                        j['transcript'] = name
                        results.append(j)
            except Exception as e:
                print(f"[Error][Extraction] {name}: {e}")
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
        path = os.path.join(self.out, 'journeys.jsonl')
        with open(path, 'w', encoding='utf-8') as f:
            for j in journeys:
                f.write(json.dumps(j) + "\n")
        print(f"JSONL saved: {path}")

    def export_csv(self, journeys: List[dict]):
        df = pd.DataFrame(journeys)
        path = os.path.join(self.out, 'journeys.csv')
        df.to_csv(path, index=False)
        print(f"CSV saved: {path}")

    def export_neo4j(self, journeys: List[dict]):
        # nodes
        nodes = set()
        for j in journeys:
            nodes.add(j['from_location']); nodes.add(j['to_location'])
        nodes_df = pd.DataFrame([{'id': n, 'label': n} for n in nodes])
        nodes_path = os.path.join(self.out, 'neo4j_nodes.csv')
        nodes_df.to_csv(nodes_path, index=False)
        # edges
        edges = []
        for j in journeys:
            edges.append({
                'source': j['from_location'],
                'target': j['to_location'],
                'date': j['approx_date'],
                'transport': j['mode_of_transport'],
                'reason': j['reason']
            })
        edges_df = pd.DataFrame(edges)
        edges_path = os.path.join(self.out, 'neo4j_edges.csv')
        edges_df.to_csv(edges_path, index=False)
        print(f"Neo4j CSVs saved: {nodes_path}, {edges_path}")

    def export_pyvis(self, journeys: List[dict]):
        net = Network(directed=True, height='750px', width='100%', notebook=False)
        net.barnes_hut()
        seen = set()
        for j in journeys:
            f, t = j['from_location'], j['to_location']
            lat1, lon1 = j['source_lat'], j['source_lon']
            lat2, lon2 = j['target_lat'], j['target_lon']
            if f not in seen:
                net.add_node(f, label=f, title=f)
                seen.add(f)
            if t not in seen:
                net.add_node(t, label=t, title=t)
                seen.add(t)
            net.add_edge(f, t, title=j['approx_date'], label=j['mode_of_transport'])
        path = os.path.join(self.out, 'journey_network.html')
        net.show(path)
        print(f"Pyvis network saved: {path}")

    def export_folium(self, journeys: List[dict]):
        # center map
        valid = [j for j in journeys if j['source_lat'] and j['target_lat']]
        center = valid[0]['source_lat'], valid[0]['source_lon'] if valid else (0, 0)
        m = folium.Map(location=center, zoom_start=5)
        for j in valid:
            pts = [(j['source_lat'], j['source_lon']), (j['target_lat'], j['target_lon'])]
            folium.PolyLine(pts, tooltip=j['approx_date']).add_to(m)
            folium.Marker(pts[0], popup=f"From: {j['from_location']}<br>{j['approx_date']} ({j['mode_of_transport']})").add_to(m)
            folium.Marker(pts[1], popup=f"To: {j['to_location']}<br>{j['reason']}").add_to(m)
        path = os.path.join(self.out, 'journey_map.html')
        m.save(path)
        print(f"Folium map saved: {path}")

    def run_all(self, transcripts: Dict[str, str]):
        journeys = self.extract(transcripts)
        journeys = self.geocode_all(journeys)
        self.export_jsonl(journeys)
        self.export_csv(journeys)
        self.export_neo4j(journeys)
        self.export_pyvis(journeys)
        self.export_folium(journeys)

# -- CLI & Streamlit UI --------------------------------------------------------
def cli_main():
    parser = argparse.ArgumentParser(description="Extract, geocode, and export journeys.")
    parser.add_argument('--input', '-i', required=True, help="Path to .txt file or folder.")
    parser.add_argument('--output', '-o', required=True, help="Output directory.")
    parser.add_argument('--model', '-M', default="gpt-4-turbo", help="OpenAI model to use.")
    parser.add_argument('--api_key', help="OpenAI API key (or set via environment).")
    args = parser.parse_args()
    api_key = get_api_key(args.api_key)
    trans = load_transcripts(args.input)
    ext = JourneyExtractor(model=args.model, api_key=api_key, output_dir=args.output)
    ext.run_all(trans)

# Streamlit UI
try:
    import streamlit as st
    from streamlit_folium import folium_static

    def ui_main():
        st.title("Historical Journey Extractor")
        uploaded = st.file_uploader("Upload .txt transcripts", type=['txt'], accept_multiple_files=True)
        out_dir = st.text_input("Output directory", value="output")
        model = st.selectbox("OpenAI model", ['gpt-4-turbo', 'gpt-3.5-turbo'])
        api_key = st.text_input("OpenAI API Key", type="password")
        if st.button("Run Extraction") and uploaded:
            os.makedirs(out_dir, exist_ok=True)
            texts = {}
            for uf in uploaded:
                texts[uf.name] = uf.read().decode('utf-8')
            ext = JourneyExtractor(model=model, api_key=get_api_key(api_key), output_dir=out_dir)
            with st.spinner("Extracting..."):
                journeys = ext.extract(texts)
                journeys = ext.geocode_all(journeys)
            st.success(f"Extracted {len(journeys)} journeys.")
            df = pd.DataFrame(journeys)
            st.dataframe(df)
            st.subheader("Folium Map")
            m = folium.Map(location=[df.loc[0,'source_lat'], df.loc[0,'source_lon']], zoom_start=5)
            for _, row in df.dropna(subset=['source_lat']).iterrows():
                pts = [(row['source_lat'], row['source_lon']), (row['target_lat'], row['target_lon'])]
                folium.PolyLine(pts, tooltip=row['approx_date']).add_to(m)
            folium_static(m)

except ImportError:
    ui_main = None

if __name__ == '__main__':
    # Detect if running via Streamlit
    if 'streamlit' in os.getenv('RUN_BY_STREAMLIT', '').lower():
        ui_main()
    elif ui_main and os.getenv('STREAMLIT_RUN'):  # another check
        ui_main()
    else:
        cli_main()
