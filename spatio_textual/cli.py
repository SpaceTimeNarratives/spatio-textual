#!/usr/bin/env python3
"""
CLI for spaCy/PlaceNames Annotator with optional multiprocessing, chunking, list-of-texts input, and --info.

Examples
--------
# 1) Quick environment check (no inputs required)
python cli.py --info --spacy-model en_core_web_trf --resources-dir resources/

# 2) Single file -> pretty JSON to stdout (chunked by default into 100 segments)
python cli.py -i data/one.txt --pretty

# 3) Batch annotate a folder (recursive) -> JSON file (array)
python cli.py -i data/ --glob "*.txt" -o out/batch.json --output-format json

# 4) Mixed inputs (file + dir + glob) with multiprocessing -> JSONL
python cli.py -i data/a.txt data/ "corpus/**/*.md" -o out/all.jsonl --workers 6 --chunksize 16 --progress --output-format jsonl

# 5) Dry run to list which files would be processed
python cli.py -i "notes/**/*.md" --glob "*.md" --dry-run

# 6) Annotate an in-memory list of segments from a JSON file (array)
#    Accepts strings or {"text": "...", "fileId": "...", "segId": N}
python cli.py --segments-json segs.json -o out/segs.json --output-format json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence, Union

from multiprocessing import get_context
import spacy  # for --info output

# Import from your utils.py (must be in the same package or PYTHONPATH)
from utils import load_spacy_model, Annotator, split_into_segments

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate one or many text files (or in-memory segments) using spaCy + PlaceNames."
    )
    # --info mode (no inputs required)
    parser.add_argument(
        "--info", action="store_true",
        help="Print annotator/model/resources info as JSON and exit."
    )
    # Inputs (optional when --info or --segments-json is used)
    parser.add_argument(
        "-i", "--input", nargs="+",
        help="File(s), directory(ies), or glob pattern(s). Examples: file.txt dir/ 'data/**/*.txt'"
    )
    parser.add_argument(
        "--glob", default="*.txt",
        help="Glob used when scanning directories. Default: *.txt",
    )
    parser.add_argument(
        "--no-recursive", action="store_true",
        help="Do NOT recurse when scanning directories.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List resolved files and exit without annotating.",
    )

    # Output
    parser.add_argument(
        "-o", "--output", default="-",
        help="Output path for JSON or JSONL. Use '-' (default) for stdout.",
    )
    parser.add_argument(
        "--output-format", choices=["json", "jsonl"], default="json",
        help="Output format: single JSON array or JSONL stream. Default: json."
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print JSON when writing a single file to stdout.",
    )
    parser.add_argument(
        "--progress", action="store_true",
        help="Show a simple progress counter while processing files.",
    )
    parser.add_argument(
        "--include-text", action="store_true",
        help="Include full source text (or segment text) in each JSON object.",
    )

    # I/O behavior
    parser.add_argument(
        "--encoding", default="utf-8",
        help="File encoding when reading inputs. Default: utf-8",
    )
    parser.add_argument(
        "--errors", default="ignore",
        choices=["strict", "ignore", "replace", "backslashreplace"],
        help="How to handle decoding errors. Default: ignore",
    )

    # Model / resources
    parser.add_argument(
        "--spacy-model",
        default=os.environ.get("SPACY_MODEL", "en_core_web_trf"),
        help="spaCy model to load (default: en_core_web_trf or $SPACY_MODEL).",
    )
    parser.add_argument(
        "--resources-dir", default=None,
        help="Optional path to the 'resources' directory (controls EntityRuler patterns and lists).",
    )
    parser.add_argument(
        "--no-entity-ruler", action="store_true",
        help="Disable the EntityRuler for this run (speed tests, etc.).",
    )

    # Chunking controls
    parser.add_argument(
        "--segments", type=int, default=100,
        help="Chunk each input file into ~N sentence-safe segments (default: 100).",
    )
    parser.add_argument(
        "--no-chunk", action="store_true",
        help="Disable chunking; annotate whole files as-is.",
    )

    # In-memory list of texts/segments
    parser.add_argument(
        "--segments-json", default=None,
        help="Path to a JSON file containing a list of texts (or objects with 'text', optional 'fileId'/'segId').",
    )

    # Parallelism
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of worker processes (>=1). Each worker loads its own spaCy pipeline. Default: 1",
    )
    parser.add_argument(
        "--chunksize", type:int, default=1,
        help="How many files each worker takes per task. Larger values reduce overhead. Default: 1",
    )

    return parser.parse_args()


def _resolve_files(
    inputs: Union[str, Path, Sequence[Union[str, Path]]],
    glob_pattern: str,
    recursive: bool,
) -> List[Path]:
    # Reuse Annotator's resolver for consistent behavior
    return list(Annotator._resolve_input_files(inputs, glob_pattern, recursive))


def _open_out(path_str: str):
    if path_str == "-" or path_str.strip() == "":
        return sys.stdout
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


# ---------- Multiprocessing worker setup ----------

# Globals inside each worker process
_WORKER = {
    "annotator": None,
    "encoding": "utf-8",
    "errors": "ignore",
    "include_text": False,
    "segments": 100,
    "no_chunk": False,
}

def _init_pool(spacy_model: str, resources_dir: str | None, add_entity_ruler: bool,
               encoding: str, errors: str, include_text: bool,
               segments: int, no_chunk: bool):
    """
    Initializer runs once per worker process. It loads spaCy + Annotator into a process-global.
    """
    global _WORKER
    nlp = load_spacy_model(spacy_model, resources_dir=resources_dir, add_entity_ruler=add_entity_ruler)
    _WORKER["annotator"] = Annotator(nlp, resources_dir=resources_dir)
    _WORKER["encoding"] = encoding
    _WORKER["errors"] = errors
    _WORKER["include_text"] = include_text
    _WORKER["segments"] = segments
    _WORKER["no_chunk"] = no_chunk

def _process_path(path_str: str) -> dict:
    """
    Map function executed in workers. Returns either:
      {"ok": True, "results": <list_of_annotation_dicts>}  # list (chunked) or [single]
    or
      {"ok": False, "file": "<path>", "error": "<repr(e)>"}
    """
    p = Path(path_str)
    try:
        if _WORKER["no_chunk"]():
            r = _WORKER["annotator"].annotate_file(
                p,
                encoding=_WORKER["encoding"],
                errors=_WORKER["errors"],
                include_text=_WORKER["include_text"],
            )
            results = [r]
        else:
            results = _WORKER["annotator"].annotate_file_chunked(
                p,
                n_segments=_WORKER["segments"],
                encoding=_WORKER["encoding"],
                errors=_WORKER["errors"],
                include_text=_WORKER["include_text"],
            )
        return {"ok": True, "results": results}
    except Exception as e:
        return {"ok": False, "file": str(p), "error": repr(e)}


def _print_info(spacy_model: str, resources_dir: str | None, add_entity_ruler: bool) -> int:
    """
    Build a transient annotator and print its environment as JSON.
    Avoids importing any lazy singletons from other modules to keep it predictable.
    """
    try:
        nlp = load_spacy_model(spacy_model, resources_dir=resources_dir, add_entity_ruler=add_entity_ruler)
        ann = Annotator(nlp, resources_dir=resources_dir)
        info = {
            "spacy_version": spacy.__version__,
            "lang": getattr(nlp, "lang", None),
            "model_name": (getattr(nlp, "meta", {}) or {}).get("name"),
            "model_meta_version": (getattr(nlp, "meta", {}) or {}).get("version"),
            "pipes": list(nlp.pipe_names),
            "has_entity_ruler": "entity_ruler" in nlp.pipe_names,
            "resources_dir": str(getattr(ann, "resources_dir", "")),
            "resource_counts": {
                "geonouns": len(getattr(ann, "geonouns", [])),
                "camps": len(getattr(ann, "camps", [])),
                "ambiguous_cities": len(getattr(ann, "ambiguous_cities", [])),
                "non_verbals": len(getattr(ann, "non_verbal", [])),
                "family_terms": len(getattr(ann, "family", [])),
            },
        }
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0
    except Exception as e:
        print(f"[ERROR] --info failed: {e}", file=sys.stderr)
        return 2


def main() -> int:
    args = parse_args()

    # --info mode (no inputs required)
    if args.info:
        return _print_info(
            spacy_model=args.spacy_model,
            resources_dir=args.resources_dir,
            add_entity_ruler=(not args.no_entity_ruler),
        )

    list_mode = args.segments_json is not None

    # Enforce inputs when not in --info and not list_mode
    if not list_mode and not args.input:
        print("error: -i/--input is required unless --info or --segments-json is used", file=sys.stderr)
        return 1

    # Handle list-of-texts mode (single-process)
    if list_mode:
        try:
            with open(args.segments_json, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as e:
            print(f"[ERROR] Failed to read --segments-json: {e}", file=sys.stderr)
            return 1

        if not isinstance(payload, list):
            print("[ERROR] --segments-json must be a JSON array", file=sys.stderr)
            return 1

        # Prepare items as dicts with text/fileId/segId
        items = []
        for i, item in enumerate(payload, 1):
            if isinstance(item, str):
                items.append({"text": item, "fileId": None, "segId": i})
            elif isinstance(item, dict) and "text" in item:
                items.append({
                    "text": item["text"],
                    "fileId": item.get("fileId"),
                    "segId": item.get("segId", i),
                })
            else:
                print("[ERROR] segments-json list must contain strings or {'text': ...} objects", file=sys.stderr)
                return 1

        # Build annotator (single-process)
        try:
            nlp = load_spacy_model(args.spacy_model, resources_dir=args.resources_dir,
                                   add_entity_ruler=(not args.no_entity_ruler))
        except Exception as e:
            print(f"Failed to load spaCy model '{args.spacy_model}': {e}", file=sys.stderr)
            return 2
        annotator = Annotator(nlp, resources_dir=args.resources_dir)

        # Group by fileId to reset segId per file
        groups = defaultdict(list)
        for it in items:
            groups[it["fileId"]].append(it)

        results = []
        for fid, group in groups.items():
            texts = [g["text"] for g in group]
            ann_results = annotator.annotate_texts(texts, file_id=fid, start_seg_id=1, include_text=args.include_text)
            # Override segId if provided in input
            for r, g in zip(ann_results, group):
                if "segId" in g and g["segId"] is not None:
                    r["segId"] = g["segId"]
            results.extend(ann_results)

        # Write output
        with _open_out(args.output) as out_f:
            if args.output_format == "jsonl":
                for r in results:
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            else:
                json.dump(results, out_f, ensure_ascii=False, indent=2)
        return 0

    # File mode: resolve files
    files = _resolve_files(
        inputs=args.input,
        glob_pattern=args.glob,
        recursive=not args.no_recursive,
    )

    if args.dry_run:
        if not files:
            print("No files matched.", file=sys.stderr)
            return 1
        for f in files:
            print(str(f))
        return 0

    if not files:
        print("No files matched. Check your -i / --input and --glob.", file=sys.stderr)
        return 1

    # Output stream selection
    to_stdout = args.output == "-" or args.output.strip() == ""
    pretty_single = args.pretty and len(files) == 1 and to_stdout

    # Pretty single-file mode stays single-process for clean stdout formatting
    if pretty_single:
        try:
            nlp = load_spacy_model(args.spacy_model, resources_dir=args.resources_dir,
                                   add_entity_ruler=(not args.no_entity_ruler))
        except Exception as e:
            print(f"Failed to load spaCy model '{args.spacy_model}': {e}", file=sys.stderr)
            return 2
        annotator = Annotator(nlp, resources_dir=args.resources_dir)
        try:
            if args.no_chunk:
                results = [annotator.annotate_file(
                    files[0],
                    encoding=args.encoding,
                    errors=args.errors,
                    include_text=args.include_text,
                )]
            else:
                results = annotator.annotate_file_chunked(
                    files[0],
                    n_segments=args.segments,
                    encoding=args.encoding,
                    errors=args.errors,
                    include_text=args.include_text,
                )
            # Pretty print list or single
            if args.output_format == "jsonl":
                # Even in pretty mode, JSONL means one record per line (not indented)
                for r in results:
                    sys.stdout.write(json.dumps(r, ensure_ascii=False) + "\n")
            else:
                json.dump(results if len(results) > 1 else results[0], sys.stdout, ensure_ascii=False, indent=2)
                sys.stdout.write("\n")
            return 0
        except Exception as e:
            print(f"[ERROR] {files[0]}: {e}", file=sys.stderr)
            return 3

    # JSON/JSONL streaming mode (stdout or file), optionally parallel
    wrote = 0
    errs = 0
    results_buffer = []  # for JSON array output
    with _open_out(args.output) as out_f:
        if args.workers <= 1:
            # Single-process path (no pool)
            try:
                nlp = load_spacy_model(args.spacy_model, resources_dir=args.resources_dir,
                                       add_entity_ruler=(not args.no_entity_ruler))
            except Exception as e:
                print(f"Failed to load spaCy model '{args.spacy_model}': {e}", file=sys.stderr)
                return 2
            annotator = Annotator(nlp, resources_dir=args.resources_dir)
            total = len(files)
            for idx, f in enumerate(files, start=1):
                if args.progress:
                    print(f"[{idx}/{total}] {f}", file=sys.stderr)
                try:
                    if args.no_chunk:
                        r_list = [annotator.annotate_file(
                            f, encoding=args.encoding, errors=args.errors, include_text=args.include_text
                        )]
                    else:
                        r_list = annotator.annotate_file_chunked(
                            f, n_segments=args.segments,
                            encoding=args.encoding, errors=args.errors, include_text=args.include_text
                        )
                    if args.output_format == "jsonl":
                        for r in r_list:
                            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                            wrote += 1
                    else:
                        results_buffer.extend(r_list)
                        wrote += len(r_list)
                except Exception as e:
                    print(f"[ERROR] {f}: {e}", file=sys.stderr)
                    errs += 1
        else:
            # Multiprocessing path (spawn for cross-platform compatibility)
            ctx = get_context("spawn")
            init_args = (
                args.spacy_model,
                args.resources_dir,
                (not args.no_entity_ruler),
                args.encoding,
                args.errors,
                args.include_text,
                args.segments,
                args.no_chunk,
            )
            with ctx.Pool(processes=args.workers, initializer=_init_pool, initargs=init_args) as pool:
                total = len(files)
                for idx, res in enumerate(
                    pool.imap_unordered(_process_path, map(str, files), chunksize=max(1, args.chunksize)),
                    start=1,
                ):
                    if args.progress:
                        print(f"[{idx}/{total}] {'OK' if res.get('ok') else 'ERR'}", file=sys.stderr)
                    if res.get("ok"):
                        r_list = res["results"]
                        if args.output_format == "jsonl":
                            for r in r_list:
                                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                                wrote += 1
                        else:
                            results_buffer.extend(r_list)
                            wrote += len(r_list)
                    else:
                        print(f"[ERROR] {res.get('file')}: {res.get('error')}", file=sys.stderr)
                        errs += 1

                pool.close()
                pool.join()

        # If JSON array, dump once
        if args.output_format == "json":
            json.dump(results_buffer, out_f, ensure_ascii=False, indent=2)

        try:
            out_f.flush()
        except Exception:
            pass

    # Print a tiny summary to stdout if writing to a file (not stdout)
    to_stdout = args.output == "-" or args.output.strip() == ""
    if not to_stdout:
        summary = {"wrote": wrote, "errors": errs, "output": args.output}
        print(json.dumps(summary, ensure_ascii=False))

    return 0 if errs == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
