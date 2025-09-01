# import argparse
# import json
# from pathlib import Path
# from .annotate import annotate_text


# def annotate_file(input_path: Path, output_path: Path):
#     with input_path.open("r", encoding="utf-8") as infile:
#         records = [json.loads(line) for line in infile]

#     results = []
#     for record in records:
#         text = record.get("text", "")
#         result = annotate_text(text)
#         record.update(result)
#         results.append(record)

#     with output_path.open("w", encoding="utf-8") as outfile:
#         for record in results:
#             json.dump(record, outfile)
#             outfile.write("\n")


# def main():
#     parser = argparse.ArgumentParser(description="Annotate spatial entities and verbs in text records.")
#     parser.add_argument("-i", "--input", required=True, help="Input JSONL file (one text record per line)")
#     parser.add_argument("-o", "--output", required=True, help="Output file path for annotated JSONL")
#     args = parser.parse_args()

#     input_path = Path(args.input)
#     output_path = Path(args.output)

#     annotate_file(input_path, output_path)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
CLI for spaCy/PlaceNames Annotator with optional multiprocessing.

Examples
--------
# Single file -> JSONL to stdout
python cli.py -i data/alice.txt --pretty

# Directory (recursive) -> JSONL file with 4 workers
python cli.py -i data/ --glob "*.txt" -o out/batch.jsonl --workers 4 --progress

# Mixed inputs (file + dir + glob) -> JSONL to stdout
python cli.py -i data/a.txt data/ "corpus/**/*.md"

# Dry run: list files that would be processed
python cli.py -i "data/**/*.txt" --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Union

from multiprocessing import get_context

# Import from your utils.py (same folder)
from utils import load_spacy_model, Annotator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate one or many text files using spaCy + PlaceNames."
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        help="File(s), directory(ies), or glob pattern(s). "
             "Examples: file.txt dir/ 'data/**/*.txt'",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        help="Output path for JSONL. Use '-' (default) for stdout.",
    )
    parser.add_argument(
        "--glob",
        default="*.txt",
        help="Glob used when scanning directories. Default: *.txt",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do NOT recurse when scanning directories.",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include full source text in each JSON object.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding when reading inputs. Default: utf-8",
    )
    parser.add_argument(
        "--errors",
        default="ignore",
        choices=["strict", "ignore", "replace", "backslashreplace"],
        help="How to handle decoding errors. Default: ignore",
    )
    parser.add_argument(
        "--spacy-model",
        default=os.environ.get("SPACY_MODEL", "en_core_web_trf"),
        help="spaCy model to load (default: en_core_web_trf or $SPACY_MODEL).",
    )
    parser.add_argument(
        "--resources-dir",
        default=None,
        help="Optional path to the 'resources' directory (overrides default for Annotator lists).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON when writing a single file to stdout.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List resolved files and exit without annotating.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a simple progress counter while processing files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (>=1). Each worker loads its own spaCy pipeline. Default: 1",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1,
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

def _init_pool(spacy_model: str, resources_dir: str | None, encoding: str, errors: str, include_text: bool):
    """
    Initializer runs once per worker process. It loads spaCy + Annotator into a process-global.
    """
    global _WORKER
    nlp = load_spacy_model(spacy_model)
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


def main() -> int:
    args = parse_args()

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

    # Pretty single-file mode stays single-process for clean stdout formatting
    to_stdout = args.output == "-" or args.output.strip() == ""
    pretty_single = args.pretty and len(files) == 1 and to_stdout
    if pretty_single:
        try:
            nlp = load_spacy_model(args.spacy_model)
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
            if not to_stdout:
                sys.stdout.flush()
            return 0
        except Exception as e:
            print(f"[ERROR] {files[0]}: {e}", file=sys.stderr)
            return 3

    # JSONL streaming mode (stdout or file), optionally parallel
    wrote = 0
    errors = 0
    with _open_out(args.output) as out_f:
        if args.workers <= 1:
            # Single-process path (no pool)
            try:
                nlp = load_spacy_model(args.spacy_model)
            except Exception as e:
                print(f"Failed to load spaCy model '{args.spacy_model}': {e}", file=sys.stderr)
                return 2
            annotator = Annotator(nlp, resources_dir=args.resources_dir)
            for idx, f in enumerate(files, start=1):
                if args.progress:
                    print(f"[{idx}/{len(files)}] {f}", file=sys.stderr)
                try:
                    result = annotator.annotate_file(
                        f, encoding=args.encoding, errors=args.errors, include_text=args.include_text
                    )
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    wrote += 1
                except Exception as e:
                    print(f"[ERROR] {f}: {e}", file=sys.stderr)
                    errors += 1
        else:
            # Multiprocessing path
            # Use 'spawn' for cross-platform compatibility (Windows/macOS/Linux)
            ctx = get_context("spawn")
            init_args = (args.spacy_model, args.resources_dir, args.encoding, args.errors, args.include_text)
            with ctx.Pool(processes=args.workers, initializer=_init_pool, initargs=init_args) as pool:
                # Iterate results as they arrive (unordered) for better throughput
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
                        errors += 1

                # Ensure pool cleanup
                pool.close()
                pool.join()

        try:
            out_f.flush()
        except Exception:
            pass

    # Print a tiny summary to stdout if writing to a file (not stdout)
    if not to_stdout:
        summary = {"wrote": wrote, "errors": errors, "output": args.output}
        print(json.dumps(summary, ensure_ascii=False))

    return 0 if errors == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
