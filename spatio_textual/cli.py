#!/usr/bin/env python3
"""
CLI for spaCy/PlaceNames Annotator with optional multiprocessing and --info.

Examples
--------
# 1) Quick environment check (no inputs required)
python cli.py --info --spacy-model en_core_web_trf --resources-dir resources/

# 2) Single file -> pretty JSON to stdout
python cli.py -i data/one.txt --pretty

# 3) Batch annotate a folder (recursive) -> JSONL file
python cli.py -i data/ --glob "*.txt" -o out/batch.jsonl

# 4) Mixed inputs (file + dir + glob) with multiprocessing
python cli.py -i data/a.txt data/ "corpus/**/*.md" -o out/all.jsonl --workers 6 --chunksize 16 --progress

# 5) Dry run to list which files would be processed
python cli.py -i "notes/**/*.md" --glob "*.md" --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Union

from multiprocessing import get_context
import spacy  # for --info output

# Import from your utils.py (must be in the same package or PYTHONPATH)
from utils import load_spacy_model, Annotator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate one or many text files using spaCy + PlaceNames."
    )
    # --info mode (no inputs required)
    parser.add_argument(
        "--info", action="store_true",
        help="Print annotator/model/resources info as JSON and exit."
    )
    # Inputs (optional when --info is used)
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
        help="Output path for JSONL. Use '-' (default) for stdout.",
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
        help="Include full source text in each JSON object.",
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

    # Parallelism
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of worker processes (>=1). Each worker loads its own spaCy pipeline. Default: 1",
    )
    parser.add_argument(
        "--chunksize", type=int, default=1,
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
}

def _init_pool(spacy_model: str, resources_dir: str | None, add_entity_ruler: bool,
               encoding: str, errors: str, include_text: bool):
    """
    Initializer runs once per worker process. It loads spaCy + Annotator into a process-global.
    """
    global _WORKER
    nlp = load_spacy_model(spacy_model, resources_dir=resources_dir, add_entity_ruler=add_entity_ruler)
    _WORKER["annotator"] = Annotator(nlp, resources_dir=resources_dir)
    _WORKER["encoding"] = encoding
    _WORKER["errors"] = errors
    _WORKER["include_text"] = include_text

def _process_path(path_str: str) -> dict:
    """
    Map function executed in workers. Returns either:
      {"ok": True, "result": <annotation_dict>}
    or
      {"ok": False, "file": "<path>", "error": "<repr(e)>"}
    """
    p = Path(path_str)
    try:
        res = _WORKER["annotator"].annotate_file(
            p,
            encoding=_WORKER["encoding"],
            errors=_WORKER["errors"],
            include_text=_WORKER["include_text"],
        )
        return {"ok": True, "result": res}
    except Exception as e:
        return {"ok": False, "file": str(p), "error": repr(e)}


def _print_info(spacy_model: str, resources_dir: str | None, add_entity_ruler: bool) -> int:
    """
    Build a transient annotator and print its environment as JSON.
    Avoids importing any lazy singletons from annotator.py to keep it predictable.
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

    # Enforce inputs when not in --info
    if not args.input:
        print("error: -i/--input is required unless --info is used", file=sys.stderr)
        return 1

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
            result = annotator.annotate_file(
                files[0],
                encoding=args.encoding,
                errors=args.errors,
                include_text=args.include_text,
            )
            json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
            return 0
        except Exception as e:
            print(f"[ERROR] {files[0]}: {e}", file=sys.stderr)
            return 3

    # JSONL streaming mode (stdout or file), optionally parallel
    wrote = 0
    errs = 0
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
                    result = annotator.annotate_file(
                        f, encoding=args.encoding, errors=args.errors, include_text=args.include_text
                    )
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    wrote += 1
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
                        out_f.write(json.dumps(res["result"], ensure_ascii=False) + "\n")
                        wrote += 1
                    else:
                        print(f"[ERROR] {res.get('file')}: {res.get('error')}", file=sys.stderr)
                        errs += 1

                pool.close()
                pool.join()

        try:
            out_f.flush()
        except Exception:
            pass

    # Print a tiny summary to stdout if writing to a file (not stdout)
    if not to_stdout:
        summary = {"wrote": wrote, "errors": errs, "output": args.output}
        print(json.dumps(summary, ensure_ascii=False))

    return 0 if errs == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())