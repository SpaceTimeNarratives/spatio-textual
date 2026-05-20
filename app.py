from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from spatio_textual.analysis import analyze_records
from spatio_textual.emotion import EmotionAnalyzer
from spatio_textual.formats import entities_to_conll
from spatio_textual.model_registry import EMOTION_MODELS, LLM_PROVIDERS, NER_MODELS, SENTIMENT_MODELS, parse_ner_model
from spatio_textual.moe import run_builtin_moe
from spatio_textual.qa import segment_testimony
from spatio_textual.sentiment import SentimentAnalyzer
from spatio_textual.transformer_ner import HFNERAnnotator
from spatio_textual.utils import Annotator, load_spacy_model, split_into_segments
from spatio_textual.viz import build_cooccurrence, to_geojson

st.set_page_config(page_title="spatio-textual annotation platform", page_icon="🗺️", layout="wide")

st.title("🗺️ spatio-textual annotation platform")
st.caption("Spatial entity annotation, entity linking, MoE adjudication, affect analysis, event extraction, telemetry and export.")


def _model_options():
    return list(NER_MODELS.keys()) + ["hf:custom", "spacy:custom"]


with st.sidebar:
    st.header("Annotation models")
    ner_choice = st.selectbox("Primary spatial NER model", _model_options(), index=0)
    if ner_choice == "hf:custom":
        ner_model = "hf:" + st.text_input("Custom HF token-classification model", value="dslim/bert-base-NER")
    elif ner_choice == "spacy:custom":
        ner_model = "spacy:" + st.text_input("Custom spaCy pipeline", value="en_core_web_trf")
    else:
        ner_model = ner_choice
    st.caption(parse_ner_model(ner_model).description)

    resources_dir = st.text_input("Resources directory", value=str(Path("spatio_textual/resources")))
    link_places = st.checkbox("Resolve/link named places to lat/lon", value=True)
    include_events = st.checkbox("Extract narrator actions/events", value=True)
    include_verbs = st.checkbox("Also export all verbs", value=False)

    st.divider()
    st.subheader("Segmentation")
    use_testimony = st.checkbox("Q/A-aware testimony segmentation", value=True)
    segmentation_mode = st.radio("Segmentation mode", ["Character budget", "Fixed segment count"], index=0)
    if segmentation_mode == "Character budget":
        max_chars = st.number_input("Max chars per segment", value=14000, min_value=500, step=500)
        n_segments = None
    else:
        n_segments = st.number_input("Number of segments", value=20, min_value=1, step=1)
        max_chars = None
    overlap_chars = st.number_input("Sentence-safe overlap chars", value=0, min_value=0, step=100)

    st.divider()
    st.subheader("Sentiment and emotion")
    sentiment_key = st.selectbox("Sentiment backend", list(SENTIMENT_MODELS.keys()), index=0)
    sentiment_backend = sentiment_key.split(":", 1)[0]
    sentiment_model = sentiment_key.split(":", 1)[1] if sentiment_key.startswith("hf:") else None
    if sentiment_key == "llm":
        sentiment_model = st.text_input("Sentiment LLM model", value="gpt-4.1-mini")
    emotion_key = st.selectbox("Emotion backend", list(EMOTION_MODELS.keys()), index=0)
    emotion_backend = emotion_key.split(":", 1)[0]
    emotion_model = emotion_key.split(":", 1)[1] if emotion_key.startswith("hf:") else None
    if emotion_key == "llm":
        emotion_model = st.text_input("Emotion LLM model", value="gpt-4.1-mini")
    llm_provider = st.selectbox("LLM provider", LLM_PROVIDERS, index=0)
    run_interpret = st.checkbox("Interpretation", value=True)

    st.divider()
    st.subheader("Mixture of experts")
    use_moe = st.checkbox("Use MoE/adjudication for entity annotation", value=False)
    moe_models = st.multiselect("Expert models", list(NER_MODELS.keys()), default=["spacy:en_core_web_trf", "spacy:en_core_web_sm"])
    threshold = st.slider("Consensus threshold", 0.0, 1.0, 0.5, 0.05)

sample_text = """Q: Where did you live before the war?
A: We lived in Amsterdam near my mother's family. Later we were deported by train to Auschwitz. I was afraid and very cold.
Q: What happened after liberation?
A: I travelled to London and felt relief when I found my brother."""

uploaded = st.file_uploader("Upload one or more .txt files", type=["txt"], accept_multiple_files=True)
text = st.text_area("Or paste text", value=sample_text, height=220)


def _make_segments(content: str, nlp=None):
    if use_testimony:
        turns = segment_testimony(content, nlp=nlp)
        segments = [
            {"text": t.text, "segStartChar": t.seg_start_char, "segEndChar": t.seg_end_char, "segTextCharLength": len(t.text)}
            for t in turns
        ]
        metadata = [{"role": t.role, "turnId": t.turn_id, "qaPairId": t.qa_pair_id, "isQuestion": t.is_question, "isAnswer": t.is_answer} for t in turns]
        return segments, metadata
    return split_into_segments(content, n_segments=int(n_segments) if n_segments else None, max_chars=int(max_chars) if max_chars else None, overlap_chars=int(overlap_chars), nlp=nlp, as_records=True), None


def _annotate_primary(file_id: str, content: str):
    spec = parse_ner_model(ner_model)
    if spec.backend == "hf":
        segments, metadata = _make_segments(content, None)
        ann = HFNERAnnotator(spec.model, link_places=link_places)
        recs = []
        for i, seg in enumerate(segments, start=1):
            rec = ann.annotate(seg["text"], include_text=True)
            rec.update({"file": file_id, "fileId": file_id, "segId": i, "segCount": len(segments), **{k: seg.get(k) for k in ("segStartChar", "segEndChar", "segTextCharLength")}})
            if metadata:
                rec.update(metadata[i - 1])
            recs.append(rec)
        return recs, segments
    nlp = load_spacy_model(spec.model, resources_dir=resources_dir)
    segments, metadata = _make_segments(content, nlp)
    annotator = Annotator(nlp, resources_dir=resources_dir, model_name=spec.model, link_places=link_places)
    recs = annotator.annotate_texts(segments, file_id=file_id, include_text=True, include_verbs=include_verbs, include_events=include_events, metadata=metadata)
    return recs, segments


if st.button("Annotate", type="primary"):
    records = []
    adjudications = []
    source_items = []
    if uploaded:
        for file in uploaded:
            source_items.append((Path(file.name).stem, file.read().decode("utf-8", errors="ignore")))
    elif text.strip():
        source_items.append(("pasted_text", text))

    for file_id, content in source_items:
        if use_moe and moe_models:
            moe = run_builtin_moe(content, moe_models, resources_dir=resources_dir, threshold=threshold, link_places=link_places)
            rec = moe.consensus
            rec.update({"file": file_id, "fileId": file_id, "segId": 1, "segCount": 1, "segStartChar": 0, "segEndChar": len(content), "segTextCharLength": len(content), "text": content, "moe_disagreements": moe.disagreements})
            records.append(rec)
            adjudications.append({"fileId": file_id, "disagreements": moe.disagreements, "requires_review": moe.requires_review})
            segments = [{"text": content}]
        else:
            recs, segments = _annotate_primary(file_id, content)
            records.extend(recs)

    texts = [r.get("text", "") for r in records]
    if sentiment_key != "none":
        sent = SentimentAnalyzer(sentiment_backend, model_name=sentiment_model, provider=llm_provider)
        for r, pred in zip(records, sent.predict(texts)):
            r["sentiment_label"] = pred.get("label")
            r["sentiment_score"] = pred.get("score")
            r["sentiment_distribution"] = pred.get("distribution")
            if pred.get("telemetry"):
                r.setdefault("telemetry", []).append(pred["telemetry"])
    if emotion_key != "none":
        emo = EmotionAnalyzer(emotion_backend, model_name=emotion_model, provider=llm_provider)
        for r, pred in zip(records, emo.predict(texts)):
            r["emotion_label"] = pred.get("label")
            r["emotion_score"] = pred.get("score")
            r["emotion_dist"] = pred.get("distribution")
            if pred.get("telemetry"):
                r.setdefault("telemetry", []).append(pred["telemetry"])
    if run_interpret:
        records = analyze_records(records)

    st.session_state["records"] = records
    st.session_state["adjudications"] = adjudications

records = st.session_state.get("records", [])
if records:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Records", "Entities", "Review queue", "Telemetry", "Visualisation", "Export"])
    with tab1:
        display = []
        for r in records:
            display.append({k: r.get(k) for k in ["fileId", "segId", "role", "isQuestion", "isAnswer", "sentiment_label", "emotion_label", "requires_review", "summary"]})
        st.dataframe(pd.DataFrame(display), use_container_width=True)
    with tab2:
        rows = []
        for r in records:
            for ent in r.get("entities") or []:
                rows.append({"fileId": r.get("fileId"), "segId": r.get("segId"), **ent})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    with tab3:
        review_rows = [r for r in records if r.get("requires_review")]
        st.write(f"Segments requiring review: {len(review_rows)}")
        for r in review_rows:
            st.markdown(f"**{r.get('fileId')} / segment {r.get('segId')}**")
            st.write(r.get("review_notes"))
            if r.get("moe_disagreements"):
                st.json(r.get("moe_disagreements"))
    with tab4:
        tel_rows = []
        for r in records:
            for tel in r.get("telemetry") or []:
                tel_rows.append({"fileId": r.get("fileId"), "segId": r.get("segId"), **tel})
        st.dataframe(pd.DataFrame(tel_rows), use_container_width=True)
        if tel_rows:
            df = pd.DataFrame(tel_rows)
            st.metric("Total estimated cost", f"${df.get('cost_usd_est', pd.Series(dtype=float)).fillna(0).sum():.4f}")
            st.metric("Mean latency", f"{df.get('latency_ms', pd.Series(dtype=float)).fillna(0).mean():.1f} ms")
    with tab5:
        edges = build_cooccurrence(records)
        st.write("Co-occurrence edges")
        st.dataframe(pd.DataFrame(edges, columns=["u", "v", "w"]), use_container_width=True)
        geojson = to_geojson(records)
        st.write(f"GeoJSON features with coordinates: {len(geojson.get('features', []))}")
        st.json(geojson)
    with tab6:
        json_bytes = json.dumps(records, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button("Download JSON", json_bytes, "spatio_textual_annotations.json", "application/json")
        jsonl = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")
        st.download_button("Download JSONL", jsonl, "spatio_textual_annotations.jsonl", "application/x-ndjson")
        csv_df = pd.DataFrame(records)
        st.download_button("Download CSV", csv_df.to_csv(index=False).encode("utf-8"), "spatio_textual_annotations.csv", "text/csv")
        conll_docs = []
        for r in records:
            tokens = (r.get("text") or "").split()
            conll_docs.append(entities_to_conll(tokens, r.get("entities") or [], doc_id=f"{r.get('fileId')}_{r.get('segId')}"))
        st.download_button("Download CoNLL/BIO", "\n".join(conll_docs).encode("utf-8"), "spatio_textual_entities.conll", "text/plain")
else:
    st.info("Paste text or upload files, then click Annotate.")
