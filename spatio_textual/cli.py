#!/usr/bin/env python3
"""
CLI for spaCy/PlaceNames Annotator with optional multiprocessing, chunking, list-of-texts input,
entity/verb toggles, multi-format saving, and optional tqdm progress bars.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Sequence, Union

from multiprocessing import get_context
import spacy  # for --info output

# Optional tqdm (progress bars)
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # gracefully degrade if not installed

from .utils import load_spacy_model, Annotator, save_annotations


# -------------------------
# Argument parsing
# -------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate text with spaCy + PlaceNames (chunking, lists, multiprocessing)."
    )
    # Info
    parser.add_argument("--info", action="store_true", help="Print annotator/model/resources info as JSON and exit.")

    # Inputs
    parser.add_argument("-i", "--input", nargs="+", help="File(s), directory(ies), or glob pattern(s).")
    parser.add_argument("--glob", default="*.txt", help="Glob used when scanning directories. Default: *.txt")
    parser.add_argument("--no-recursive", action="store_true", help="Do NOT recurse when scanning directories.")
    parser.add_argument("--dry-run", action="store_true", help="List resolved files and exit without annotating.")

    # Output
    parser.add_argument("-o", "--output", default="-", help="Output path. '-' (default) for stdout.")
    parser.add_argument(
        "--output-format",
        choices=["json", "jsonl", "tsv", "csv"],
        default="json",
        help="Output format (default: json).",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON to stdout when feasible.")
    parser.add_argument("--progress", action="store_true", help="Print a simple [idx/total] progress line to stderr.")
    parser.add_argument("--tqdm", action="store_true", help="Use a tqdm progress bar (requires tqdm). Overrides --progress.")
    parser.add_argument("--include-text", action="store_true", help="Include full text in each JSON object.")

    # I/O behavior
    parser.add_argument("--encoding", default="utf-8", help="File encoding for inputs. Default: utf-8")
    parser.add_argument(
        "--errors",
        default="ignore",
        choices=["strict", "ignore", "replace", "backslashreplace"],
        help="How to handle decoding errors. Default: ignore",
    )

    # Model / resources
    parser.add_argument(
        "--spacy-model",
        default=os.environ.get("SPACY_MODEL", "en_core_web_trf"),
        help="spaCy model to load (default: en_core_web_trf or $SPACY_MODEL).",
    )
    parser.add_argument("--resources-dir", default=None, help="Path to resources (patterns and lists).")
    parser.add_argument("--no-entity-ruler", action="store_true", help="Disable the EntityRuler.")

    # Chunking
    parser.add_argument("--segments", type=int, default=100, help="Chunk each input file into ~N sentence-safe segments.")
    parser.add_argument("--no-chunk", action="store_true", help="Disable chunking; annotate whole files.")

    # In-memory list
    parser.add_argument("--segments-json", default=None, help="JSON file with a list of texts/segments.")

    # Optional toggles
    parser.add_argument("--no-entities", action="store_true", help="Disable entity annotation.")
    parser.add_argument("--verbs", action="store_true", help="Enable verb extraction.")

    # Parallelism
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (>=1).")
    parser.add_argument("--chunksize", type=int, default=1, help="Files per task for workers.")

    return parser.parse_args()


# -------------------------
# Helpers
# -------------------------

def _resolve_files(inputs: Union[str, Path, Sequence[Union[str, Path]]], glob_pattern: str, recursive: bool) -> List[Path]:
    return list(Annotator._resolve_input_files(inputs, glob_pattern, recursive))


def _open_out(path_str: str):
    if path_str == "-" or path_str.strip() == "":
        return sys.stdout
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


# -------------------------
# Multiprocessing worker setup
# -------------------------
_WORKER = {
    "annotator": None,
    "encoding": "utf-8",
    "errors": "ignore",
    "include_text": False,
    "segments": 100,
    "no_chunk": False,
    "include_entities": True,
    "include_verbs": False,
}


def _init_pool(spacy_model: str, resources_dir: str | None, add_entity_ruler: bool,
               encoding: str, errors: str, include_text: bool,
               segments: int, no_chunk: bool, include_entities: bool, include_verbs: bool):
    global _WORKER
    nlp = load_spacy_model(spacy_model, resources_dir=resources_dir, add_entity_ruler=add_entity_ruler)
    _WORKER["annotator"] = Annotator(nlp, resources_dir=resources_dir)
    _WORKER["encoding"] = encoding
    _WORKER["errors"] = errors
    _WORKER["include_text"] = include_text
    _WORKER["segments"] = segments
    _WORKER["no_chunk"] = no_chunk
    _WORKER["include_entities"] = include_entities
    _WORKER["include_verbs"] = include_verbs


def _process_path(path_str: str) -> dict:
    p = Path(path_str)
    try:
        if _WORKER["no_chunk"]:
            r = _WORKER["annotator"].annotate_file(
                p,
                encoding=_WORKER["encoding"],
                errors=_WORKER["errors"],
                include_text=_WORKER["include_text"],
                include_entities=_WORKER["include_entities"],
                include_verbs=_WORKER["include_verbs"],
            )
            results = [r]
        else:
            results = _WORKER["annotator"].annotate_file_chunked(
                p,
                n_segments=_WORKER["segments"],
                encoding=_WORKER["encoding"],
                errors=_WORKER["errors"],
                include_text=_WORKER["include_text"],
                include_entities=_WORKER["include_entities"],
                include_verbs=_WORKER["include_verbs"],
            )
        return {"ok": True, "results": results}
    except Exception as e:
        return {"ok": False, "file": str(p), "error": repr(e)}


# -------------------------
# Info printer
# -------------------------

def _print_info(spacy_model: str, resources_dir: str | None, add_entity_ruler: bool) -> int:
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


# -------------------------
# Main
# -------------------------

def main() -> int:
    args = parse_args()

    # --info
    if args.info:
        return _print_info(
            spacy_model=args.spacy_model,
            resources_dir=args.resources_dir,
            add_entity_ruler=(not args.no_entity_ruler),
        )

    list_mode = args.segments_json is not None

    if not list_mode and not args.input:
        print("error: -i/--input is required unless --info or --segments-json is used", file=sys.stderr)
        return 1

    include_entities = (not args.no_entities)
    include_verbs = args.verbs

    # ---------- list-of-texts mode (single process) ----------
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

        items = []
        for i, item in enumerate(payload, 1):
            if isinstance(item, str):
                items.append({"text": item, "fileId": None, "segId": i})
            elif isinstance(item, dict) and "text" in item:
                items.append({"text": item["text"], "fileId": item.get("fileId"), "segId": item.get("segId", i)})
            else:
                print("[ERROR] segments-json list must contain strings or {'text': ...} objects", file=sys.stderr)
                return 1

        try:
            nlp = load_spacy_model(args.spacy_model, resources_dir=args.resources_dir,
                                   add_entity_ruler=(not args.no_entity_ruler))
        except Exception as e:
            print(f"Failed to load spaCy model '{args.spacy_model}': {e}", file=sys.stderr)
            return 2
        annotator = Annotator(nlp, resources_dir=args.resources_dir)

        groups = defaultdict(list)
        for it in items:
            groups[it["fileId"]].append(it)

        results: List[dict] = []
        for fid, group in groups.items():
            texts = [g["text"] for g in group]
            ann_results = annotator.annotate_texts(
                texts, file_id=fid, start_seg_id=1, include_text=args.include_text,
                include_entities=include_entities, include_verbs=include_verbs
            )
            for r, g in zip(ann_results, group):
                if "segId" in g and g["segId"] is not None:
                    r["segId"] = g["segId"]
            results.extend(ann_results)

        # Save
        if args.output == "-" and args.output_format == "jsonl":
            for r in results:
                sys.stdout.write(json.dumps(r, ensure_ascii=False) + "\n")
        elif args.output == "-" and args.output_format == "json":
            json.dump(results, sys.stdout, ensure_ascii=False, indent=2); sys.stdout.write("\n")
        else:
            save_annotations(results, args.output, fmt=args.output_format)
            print(json.dumps({"wrote": len(results), "errors": 0, "output": args.output}, ensure_ascii=False))
        return 0

    # ---------- file mode ----------
    files = _resolve_files(inputs=args.input, glob_pattern=args.glob, recursive=not args.no_recursive)

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

    to_stdout = args.output == "-" or args.output.strip() == ""
    pretty_single = args.pretty and len(files) == 1 and to_stdout and args.output_format in {"json"}

    # pretty single-file mode (keeps it single-process for clean stdout)
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
                    encoding=args.encoding, errors=args.errors,
                    include_text=args.include_text,
                    include_entities=include_entities, include_verbs=include_verbs,
                )]
            else:
                results = annotator.annotate_file_chunked(
                    files[0], n_segments=args.segments,
                    encoding=args.encoding, errors=args.errors,
                    include_text=args.include_text,
                    include_entities=include_entities, include_verbs=include_verbs,
                )
            json.dump(results if len(results) > 1 else results[0], sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
            return 0
        except Exception as e:
            print(f"[ERROR] {files[0]}: {e}", file=sys.stderr)
            return 3

    # streaming/parallel or collected outputs
    wrote = 0
    errs = 0
    results_buffer: List[dict] = []

    # For tsv/csv we collect; for json we collect; for jsonl we can stream
    collect_only = args.output_format in {"tsv", "csv", "json"} or (args.output != "-" and args.output_format == "jsonl")

    # choose progress UI
    use_tqdm = bool(args.tqdm and tqdm is not None)
    progress_bar = None

    if args.workers <= 1:
        try:
            nlp = load_spacy_model(args.spacy_model, resources_dir=args.resources_dir,
                                   add_entity_ruler=(not args.no_entity_ruler))
        except Exception as e:
            print(f"Failed to load spaCy model '{args.spacy_model}': {e}", file=sys.stderr)
            return 2
        annotator = Annotator(nlp, resources_dir=args.resources_dir)

        total = len(files)
        if use_tqdm:
            progress_bar = tqdm(total=total, desc="Annotating", unit="file")  # type: ignore

        for idx, f in enumerate(files, start=1):
            if not use_tqdm and args.progress:
                print(f"[{idx}/{total}] {f}", file=sys.stderr)
            try:
                if args.no_chunk:
                    r_list = [annotator.annotate_file(
                        f, encoding=args.encoding, errors=args.errors, include_text=args.include_text,
                        include_entities=include_entities, include_verbs=include_verbs
                    )]
                else:
                    r_list = annotator.annotate_file_chunked(
                        f, n_segments=args.segments,
                        encoding=args.encoding, errors=args.errors, include_text=args.include_text,
                        include_entities=include_entities, include_verbs=include_verbs
                    )
                if args.output_format == "jsonl" and not collect_only:
                    with _open_out(args.output) as out_f2:
                        for r in r_list:
                            out_f2.write(json.dumps(r, ensure_ascii=False) + "\n")
                            wrote += 1
                else:
                    results_buffer.extend(r_list)
                    wrote += len(r_list)
            except Exception as e:
                print(f"[ERROR] {f}: {e}", file=sys.stderr)
                errs += 1
            finally:
                if progress_bar:
                    progress_bar.update(1)  # type: ignore
        if progress_bar:
            progress_bar.close()  # type: ignore

    else:
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
            include_entities,
            include_verbs,
        )
        with ctx.Pool(processes=args.workers, initializer=_init_pool, initargs=init_args) as pool:
            total = len(files)

            if args.output_format == "jsonl" and not collect_only:
                with _open_out(args.output) as out_f:
                    iterator = pool.imap_unordered(_process_path, map(str, files), chunksize=max(1, args.chunksize))
                    if use_tqdm:
                        progress_bar = tqdm(total=total, desc="Annotating", unit="file")  # type: ignore
                    for idx, res in enumerate(iterator, start=1):
                        if not use_tqdm and args.progress:
                            print(f"[{idx}/{total}] {'OK' if res.get('ok') else 'ERR'}", file=sys.stderr)
                        if res.get("ok"):
                            for r in res["results"]:
                                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                                wrote += 1
                        else:
                            print(f"[ERROR] {res.get('file')}: {res.get('error')}", file=sys.stderr)
                            errs += 1
                        if progress_bar:
                            progress_bar.update(1)  # type: ignore
                    if progress_bar:
                        progress_bar.close()  # type: ignore
            else:
                iterator = pool.imap_unordered(_process_path, map(str, files), chunksize=max(1, args.chunksize))
                if use_tqdm:
                    progress_bar = tqdm(total=total, desc="Annotating", unit="file")  # type: ignore
                for idx, res in enumerate(iterator, start=1):
                    if not use_tqdm and args.progress:
                        print(f"[{idx}/{total}] {'OK' if res.get('ok') else 'ERR'}", file=sys.stderr)
                    if res.get("ok"):
                        results_buffer.extend(res["results"])
                        wrote += len(res["results"])
                    else:
                        print(f"[ERROR] {res.get('file')}: {res.get('error')}", file=sys.stderr)
                        errs += 1
                    if progress_bar:
                        progress_bar.update(1)  # type: ignore
                if progress_bar:
                    progress_bar.close()  # type: ignore

            pool.close()
            pool.join()

    # Save collected results
    if collect_only:
        if args.output == "-":
            if args.output_format == "json":
                json.dump(results_buffer, sys.stdout, ensure_ascii=False, indent=2); sys.stdout.write("\n")
            elif args.output_format in {"csv", "tsv"}:
                # Not ideal for stdout; suggest a path in docs, but we still support it
                tmp_path = Path("/tmp/_stdout_tmp." + args.output_format)
                save_annotations(results_buffer, tmp_path, fmt=args.output_format)
                sys.stdout.write(tmp_path.read_text(encoding="utf-8"))
            else:  # jsonl to stdout but we collected
                for r in results_buffer:
                    sys.stdout.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            save_annotations(results_buffer, args.output, fmt=args.output_format)
            print(json.dumps({"wrote": wrote, "errors": errs, "output": args.output}, ensure_ascii=False))

    return 0 if errs == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
