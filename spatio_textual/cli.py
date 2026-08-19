from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

from .analysis import analyze_records
from .emotion import EmotionAnalyzer
from .model_registry import DEFAULT_NER_MODEL, parse_ner_model
from .moe import run_builtin_moe
from .qa import segment_testimony
from .sentiment import SentimentAnalyzer
from .transformer_ner import HFNERAnnotator
from .utils import Annotator, load_spacy_model, save_annotations, split_into_segments
from .viz import build_cooccurrence, make_map_geojson, to_geojson

_WORKER: Annotator | None = None
_WORKER_MODEL_KEY: str | None = None


def _init_worker(model_key: str, resources_dir: str | None, link_places: bool):
    global _WORKER, _WORKER_MODEL_KEY
    spec = parse_ner_model(model_key)
    if spec.backend == "hf":
        _WORKER = None
    else:
        _WORKER = Annotator(load_spacy_model(spec.model, resources_dir=resources_dir), resources_dir=resources_dir, model_name=spec.model, link_places=link_places)
    _WORKER_MODEL_KEY = model_key


def _annotate_segments_spacy(segments: list[dict[str, Any]], file_id: str, cfg: dict[str, Any], metadata: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    global _WORKER
    if _WORKER is None:
        _init_worker(cfg["ner_model"], cfg.get("resources_dir"), cfg.get("link_places", True))
    assert _WORKER is not None
    return _WORKER.annotate_texts(
        segments,
        file_id=file_id,
        include_text=cfg.get("include_text", True),
        include_entities=True,
        include_verbs=cfg.get("verbs", False),
        include_events=cfg.get("events", True),
        metadata=metadata or None,
    )


def _annotate_segments_hf(segments: list[dict[str, Any]], file_id: str, cfg: dict[str, Any], metadata: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    spec = parse_ner_model(cfg["ner_model"])
    ann = HFNERAnnotator(spec.model, link_places=cfg.get("link_places", True))
    records = []
    for idx, seg in enumerate(segments, start=1):
        rec = ann.annotate(seg.get("text", ""), include_text=cfg.get("include_text", True))
        rec.update({"file": file_id, "fileId": file_id, "segId": idx, "segCount": len(segments), **{k: seg.get(k) for k in ("segStartChar", "segEndChar", "segTextCharLength")}})
        if metadata and idx - 1 < len(metadata):
            rec.update(metadata[idx - 1])
        records.append(rec)
    return records


def _apply_affect_and_interpret(recs: list[dict[str, Any]], segments: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    texts = [r.get("text") or s.get("text", "") for r, s in zip(recs, segments)]
    if cfg.get("sentiment_backend") != "none":
        preds = SentimentAnalyzer(cfg.get("sentiment_backend", "rule"), model_name=cfg.get("sentiment_model"), provider=cfg.get("llm_provider", "openai")).predict(texts)
        for r, pred in zip(recs, preds):
            r["sentiment_label"] = pred.get("label")
            r["sentiment_score"] = pred.get("score")
            r["sentiment_distribution"] = pred.get("distribution")
            if pred.get("telemetry"):
                r.setdefault("telemetry", []).append(pred["telemetry"])
    if cfg.get("emotion_backend") != "none":
        preds = EmotionAnalyzer(cfg.get("emotion_backend", "rule"), model_name=cfg.get("emotion_model"), provider=cfg.get("llm_provider", "openai")).predict(texts)
        for r, pred in zip(recs, preds):
            r["emotion_label"] = pred.get("label")
            r["emotion_score"] = pred.get("score")
            r["emotion_dist"] = pred.get("distribution")
            if pred.get("telemetry"):
                r.setdefault("telemetry", []).append(pred["telemetry"])
    if cfg.get("interpret"):
        recs = analyze_records(recs)
    return recs


def _annotate_file(args: tuple[str, dict[str, Any]]) -> list[dict[str, Any]]:
    path, cfg = args
    p = Path(path)
    text = p.read_text(encoding=cfg.get("encoding", "utf-8"), errors=cfg.get("errors", "ignore"))
    metadata = []
    if cfg.get("testimony"):
        nlp = _WORKER.nlp if _WORKER is not None else None
        turns = segment_testimony(text, nlp=nlp)
        segments = [
            {"text": t.text, "segStartChar": t.seg_start_char, "segEndChar": t.seg_end_char, "segTextCharLength": len(t.text)}
            for t in turns
        ]
        metadata = [{"role": t.role, "turnId": t.turn_id, "qaPairId": t.qa_pair_id, "isQuestion": t.is_question, "isAnswer": t.is_answer} for t in turns]
    else:
        nlp = _WORKER.nlp if _WORKER is not None else None
        segments = split_into_segments(text, n_segments=cfg.get("n_segments"), nlp=nlp, max_chars=cfg.get("max_chars"), overlap_chars=cfg.get("overlap_chars", 0), as_records=True)
    spec = parse_ner_model(cfg["ner_model"])
    if cfg.get("moe_models"):
        # MoE currently adjudicates per whole text to maximise expert context.
        moe = run_builtin_moe(text, cfg["moe_models"], resources_dir=cfg.get("resources_dir"), threshold=cfg.get("moe_threshold", 0.5), link_places=cfg.get("link_places", True))
        recs = [moe.consensus]
        recs[0].update({"file": str(p), "fileId": p.stem, "segId": 1, "segCount": 1, "segStartChar": 0, "segEndChar": len(text), "segTextCharLength": len(text), "moe_disagreements": moe.disagreements})
    elif spec.backend == "hf":
        recs = _annotate_segments_hf(segments, p.stem, cfg, metadata)
    else:
        recs = _annotate_segments_spacy(segments, p.stem, cfg, metadata)
    return _apply_affect_and_interpret(recs, segments if len(recs) == len(segments) else [{"text": text}], cfg)


def _resolve_files(inputs: list[str], glob: str, recursive: bool) -> list[str]:
    out: list[str] = []
    for item in inputs:
        p = Path(item)
        if p.is_file():
            out.append(str(p))
        elif p.is_dir():
            pattern = f"**/{glob}" if recursive else glob
            out.extend(str(x) for x in sorted(p.glob(pattern)) if x.is_file())
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="spatio-textual", description="Spatial textual annotation for files, folders and testimony transcripts.")
    p.add_argument("-i", "--input", nargs="+", help="Input text files or directories")
    p.add_argument("--segments-json", help="JSON array of text segments instead of files")
    p.add_argument("--glob", default="*.txt")
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument("-o", "--output", default="-", help="Output path or '-' for stdout JSON")
    p.add_argument("--output-format", choices=["json", "jsonl", "csv", "tsv"], default="jsonl")
    p.add_argument("--ner-model", default=DEFAULT_NER_MODEL, help="NER model key, e.g. spacy:en_core_web_trf, spacy:en_core_web_sm, hf:dslim/bert-base-NER")
    p.add_argument("--spacy-model", help="Backwards-compatible alias for --ner-model")
    p.add_argument("--resources-dir")
    p.add_argument("--n-segments", type=int)
    p.add_argument("--max-chars", type=int, default=14000)
    p.add_argument("--overlap-chars", type=int, default=0)
    p.add_argument("--include-text", action="store_true", default=True)
    p.add_argument("--verbs", action="store_true", help="Export all detected verbs")
    p.add_argument("--events", action=argparse.BooleanOptionalAction, default=True, help="Extract narrator-centred events/actions")
    p.add_argument("--testimony", action="store_true")
    p.add_argument("--link-places", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sentiment-backend", choices=["none", "rule", "hf", "llm"], default="none")
    p.add_argument("--sentiment", choices=["rule", "none"], help="Deprecated alias for --sentiment-backend")
    p.add_argument("--sentiment-model")
    p.add_argument("--emotion-backend", choices=["none", "rule", "hf", "llm"], default="none")
    p.add_argument("--emotion", choices=["rule", "none"], help="Deprecated alias for --emotion-backend")
    p.add_argument("--emotion-model")
    p.add_argument("--llm-provider", default="openai")
    p.add_argument("--interpret", action="store_true")
    p.add_argument("--moe-models", nargs="*", help="Run MoE adjudication with model keys. Example: --moe-models spacy:en_core_web_trf hf:dslim/bert-base-NER")
    p.add_argument("--moe-threshold", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--chunksize", type=int, default=8)
    p.add_argument("--tqdm", action="store_true")
    p.add_argument("--geojson-out")
    p.add_argument("--map-out")
    p.add_argument("--cooccurrence-out")
    p.add_argument("--info", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.spacy_model:
        args.ner_model = args.spacy_model if args.spacy_model.startswith(("spacy:", "hf:")) else f"spacy:{args.spacy_model}"
    if args.sentiment:
        args.sentiment_backend = args.sentiment
    if args.emotion:
        args.emotion_backend = args.emotion
    if args.info:
        print(json.dumps({"package": "spatio-textual", "ner_model": args.ner_model, "workers": args.workers}, ensure_ascii=False, indent=2))
        return 0

    cfg = vars(args).copy()
    results: list[dict[str, Any]] = []
    if args.segments_json:
        raw = json.loads(Path(args.segments_json).read_text(encoding="utf-8"))
        segments = [{"text": str(x), "segStartChar": None, "segEndChar": None, "segTextCharLength": len(str(x))} for x in raw]
        spec = parse_ner_model(args.ner_model)
        if spec.backend == "hf":
            results = _annotate_segments_hf(segments, "segments", cfg)
        else:
            ann = Annotator(load_spacy_model(spec.model, resources_dir=args.resources_dir), resources_dir=args.resources_dir, model_name=spec.model, link_places=args.link_places)
            results = ann.annotate_texts(segments, file_id="segments", include_text=True, include_verbs=args.verbs, include_events=args.events)
        results = _apply_affect_and_interpret(results, segments, cfg)
    else:
        files = _resolve_files(args.input or [], args.glob, not args.no_recursive)
        tasks = [(f, cfg) for f in files]
        spec = parse_ner_model(args.ner_model)
        use_mp = args.workers and args.workers > 1 and len(tasks) > 1 and spec.backend != "hf" and not args.moe_models
        if use_mp:
            with mp.Pool(processes=args.workers, initializer=_init_worker, initargs=(args.ner_model, args.resources_dir, args.link_places)) as pool:
                imap = pool.imap_unordered(_annotate_file, tasks, chunksize=max(1, args.chunksize))
                if args.tqdm:
                    from tqdm import tqdm
                    imap = tqdm(imap, total=len(tasks), desc="Annotating")
                for recs in imap:
                    results.extend(recs)
        else:
            if args.tqdm:
                from tqdm import tqdm
                tasks_iter = tqdm(tasks, total=len(tasks), desc="Annotating")
            else:
                tasks_iter = tasks
            for task in tasks_iter:
                results.extend(_annotate_file(task))

    if args.geojson_out:
        geojson = to_geojson(results)
        Path(args.geojson_out).write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.map_out:
            make_map_geojson(geojson, args.map_out)
    if args.cooccurrence_out:
        edges = [{"u": u, "v": v, "w": w} for u, v, w in build_cooccurrence(results)]
        save_annotations(edges, args.cooccurrence_out, fmt="csv" if str(args.cooccurrence_out).endswith(".csv") else "tsv")

    if args.output == "-":
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        save_annotations(results, args.output, fmt=args.output_format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
