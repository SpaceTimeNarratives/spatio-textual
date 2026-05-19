from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from spatio_textual.analysis import analyze_records
from spatio_textual.emotion import EmotionAnalyzer
from spatio_textual.formats import entities_to_conll
from spatio_textual.moe import run_builtin_moe
from spatio_textual.qa import segment_testimony
from spatio_textual.sentiment import SentimentAnalyzer
from spatio_textual.utils import Annotator, load_spacy_model, split_into_segments
from spatio_textual.viz import build_cooccurrence, to_geojson

st.set_page_config(page_title="spatio-textual annotation platform", page_icon="🗺️", layout="wide")

st.title("🗺️ spatio-textual annotation platform")
st.caption("Entity annotation, Q/A-aware testimony segmentation, MoE adjudication, affect analysis, interpretation and visualisation.")

with st.sidebar:
    st.header("Configuration")
    model_name = st.text_input("spaCy model", value="en_core_web_sm")
    resources_dir = st.text_input("Resources directory", value=str(Path("spatio_textual/resources")))
    include_verbs = st.checkbox("Extract verbs", value=True)
    use_testimony = st.checkbox("Q/A-aware testimony segmentation", value=True)
    segmentation_mode = st.radio("Segmentation mode", ["Character budget", "Fixed segment count"], index=0)
    if segmentation_mode == "Character budget":
        max_chars = st.number_input("Max chars per segment", value=14000, min_value=500, step=500)
        n_segments = None
    else:
        n_segments = st.number_input("Number of segments", value=20, min_value=1, step=1)
        max_chars = None
    run_sentiment = st.checkbox("Sentiment", value=True)
    run_emotion = st.checkbox("Emotion", value=True)
    run_interpret = st.checkbox("Interpretation", value=True)
    st.divider()
    st.subheader("MoE adjudication")
    moe_models = st.multiselect("Offline annotators", ["spaCy+ruler", "spaCy"], default=["spaCy+ruler", "spaCy"])
    threshold = st.slider("Consensus threshold", 0.0, 1.0, 0.5, 0.05)

sample_text = """Q: Where did you live before the war?
A: We lived in Amsterdam near my mother's family. Later we were deported by train to Auschwitz. I was afraid and very cold.
Q: What happened after liberation?
A: I travelled to London and felt relief when I found my brother."""

uploaded = st.file_uploader("Upload one or more .txt files", type=["txt"], accept_multiple_files=True)
text = st.text_area("Or paste text", value=sample_text, height=220)

if st.button("Annotate", type="primary"):
    nlp = load_spacy_model(model_name, resources_dir=resources_dir)
    annotator = Annotator(nlp, resources_dir=resources_dir)
    records = []
    source_items = []
    if uploaded:
        for file in uploaded:
            source_items.append((Path(file.name).stem, file.read().decode("utf-8", errors="ignore")))
    elif text.strip():
        source_items.append(("pasted_text", text))

    for file_id, content in source_items:
        if use_testimony:
            turns = segment_testimony(content, nlp=nlp)
            segments = [t.text for t in turns]
            metadata = [{"role": t.role, "turnId": t.turn_id, "qaPairId": t.qa_pair_id, "isQuestion": t.is_question, "isAnswer": t.is_answer} for t in turns]
        else:
            segments = split_into_segments(content, n_segments=int(n_segments) if n_segments else None, max_chars=int(max_chars) if max_chars else None, nlp=nlp)
            metadata = None
        recs = annotator.annotate_texts(segments, file_id=file_id, include_text=True, include_verbs=include_verbs, metadata=metadata)
        records.extend(recs)

    texts = [r.get("text", "") for r in records]
    if run_sentiment:
        for r, pred in zip(records, SentimentAnalyzer("rule").predict(texts)):
            r["sentiment_label"], r["sentiment_score"] = pred["label"], pred["score"]
    if run_emotion:
        for r, pred in zip(records, EmotionAnalyzer("rule").predict(texts)):
            r["emotion_label"], r["emotion_score"], r["emotion_dist"] = pred["label"], pred["score"], pred.get("distribution")
    if run_interpret:
        records = analyze_records(records)
    st.session_state["records"] = records

    if moe_models and source_items:
        moe = run_builtin_moe(source_items[0][1], moe_models, resources_dir=resources_dir)
        st.session_state["moe"] = moe

records = st.session_state.get("records", [])
if records:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Records", "Entities", "Adjudication", "Visualisation", "Export"])
    with tab1:
        display = []
        for r in records:
            display.append({k: r.get(k) for k in ["fileId", "segId", "role", "isQuestion", "isAnswer", "sentiment_label", "emotion_label", "summary"]})
        st.dataframe(pd.DataFrame(display), use_container_width=True)
    with tab2:
        rows = []
        for r in records:
            for ent in r.get("entities") or []:
                row = {"fileId": r.get("fileId"), "segId": r.get("segId"), **ent}
                rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    with tab3:
        moe = st.session_state.get("moe")
        if moe:
            st.write("Consensus entities")
            st.dataframe(pd.DataFrame(moe.consensus.get("entities", [])), use_container_width=True)
            st.write("Disagreements")
            st.json(moe.disagreements)
        else:
            st.info("Run annotation with at least one MoE annotator selected.")
    with tab4:
        edges = build_cooccurrence(records)
        st.write("Co-occurrence edges")
        st.dataframe(pd.DataFrame(edges, columns=["u", "v", "w"]), use_container_width=True)
        geojson = to_geojson(records)
        st.write(f"GeoJSON features with coordinates: {len(geojson.get('features', []))}")
        st.json(geojson)
    with tab5:
        json_bytes = json.dumps(records, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button("Download JSON", json_bytes, "spatio_textual_annotations.json", "application/json")
        jsonl = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")
        st.download_button("Download JSONL", jsonl, "spatio_textual_annotations.jsonl", "application/x-ndjson")
        ent_rows = []
        for r in records:
            tokens = (r.get("text") or "").split()
            ent_rows.append(entities_to_conll(tokens, r.get("entities") or [], doc_id=f"{r.get('fileId')}_{r.get('segId')}"))
        st.download_button("Download CoNLL/BIO", "\n".join(ent_rows).encode("utf-8"), "spatio_textual_entities.conll", "text/plain")
else:
    st.info("Paste text or upload files, then click Annotate.")
