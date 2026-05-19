from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

from .analysis import analyze_records
from .emotion import EmotionAnalyzer
from .qa import segment_testimony
from .sentiment import SentimentAnalyzer
from .utils import Annotator, load_spacy_model, save_annotations, split_into_segments
from .viz import build_cooccurrence, make_map_geojson, to_geojson

_WORKER: Annotator | None = None


def _init_worker(model: str, resources_dir: str | None):
    global _WORKER
    _WORKER = Annotator(load_spacy_model(model, resources_dir=resources_dir), resources_dir=resources_dir)


def _annotate_file(args: tuple[str, dict[str, Any]]) -> list[dict[str, Any]]:
    path, cfg = args
    global _WORKER
    if _WORKER is None:
        _init_worker(cfg["spacy_model"], cfg.get("resources_dir"))
    p = Path(path)
    text = p.read_text(encoding=cfg.get("encoding", "utf-8"), errors=cfg.get("errors", "ignore"))
    segments = []
    metadata = []
    if cfg.get("testimony"):
        turns = segment_testimony(text, nlp=_WORKER.nlp)
        segments = [t.text for t in turns]
        metadata = [{
            "role": t.role,
            "turnId": t.turn_id,
            "qaPairId": t.qa_pair_id,
            "isQuestion": t.is_question,
            "isAnswer": t.is_answer,
        } for t in turns]
    else:
        segments = split_into_segments(text, n_segments=cfg.get("n_segments"), nlp=_WORKER.nlp, max_chars=cfg.get("max_chars"), overlap_chars=cfg.get("overlap_chars", 0))
    recs = _WORKER.annotate_texts(
        segments,
        file_id=p.stem,
        include_text=cfg.get("include_text", True),
        include_entities=True,
        include_verbs=cfg.get("verbs", False),
        metadata=metadata or None,
    )
    texts = [r.get("text") or s for r, s in zip(recs, segments)]
    if cfg.get("sentiment"):
        preds = SentimentAnalyzer("rule").predict(texts)
        for r, pred in zip(recs, preds):
            r["sentiment_label"] = pred["label"]
            r["sentiment_score"] = pred["score"]
    if cfg.get("emotion"):
        preds = EmotionAnalyzer("rule").predict(texts)
        for r, pred in zip(recs, preds):
            r["emotion_label"] = pred["label"]
            r["emotion_score"] = pred["score"]
            r["emotion_dist"] = pred.get("distribution")
    if cfg.get("interpret"):
        recs = analyze_records(recs)
    return recs


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
    p.add_argument("--spacy-model", default="en_core_web_sm")
    p.add_argument("--resources-dir")
    p.add_argument("--n-segments", type=int)
    p.add_argument("--max-chars", type=int, default=14000)
    p.add_argument("--overlap-chars", type=int, default=0)
    p.add_argument("--include-text", action="store_true", default=True)
    p.add_argument("--verbs", action="store_true")
    p.add_argument("--testimony", action="store_true")
    p.add_argument("--sentiment", choices=["rule", "none"], default="none")
    p.add_argument("--emotion", choices=["rule", "none"], default="none")
    p.add_argument("--interpret", action="store_true")
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
    if args.info:
        print(json.dumps({"package": "spatio-textual", "model": args.spacy_model, "workers": args.workers}, ensure_ascii=False, indent=2))
        return 0

    cfg = vars(args).copy()
    cfg["sentiment"] = args.sentiment == "rule"
    cfg["emotion"] = args.emotion == "rule"

    results: list[dict[str, Any]] = []
    if args.segments_json:
        texts = json.loads(Path(args.segments_json).read_text(encoding="utf-8"))
        ann = Annotator(load_spacy_model(args.spacy_model, resources_dir=args.resources_dir), resources_dir=args.resources_dir)
        results = ann.annotate_texts(texts, file_id="segments", include_text=True, include_verbs=args.verbs)
    else:
        files = _resolve_files(args.input or [], args.glob, not args.no_recursive)
        tasks = [(f, cfg) for f in files]
        if args.workers and args.workers > 1 and len(tasks) > 1:
            iterator = None
            with mp.Pool(processes=args.workers, initializer=_init_worker, initargs=(args.spacy_model, args.resources_dir)) as pool:
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
