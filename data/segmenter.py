import os
import math
import json
import argparse
import nltk
from nltk.tokenize import sent_tokenize
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging

# Download punkt if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

"""
Split text into exactly N segments per file, but if a file has fewer sentences than N,
use one segment per sentence (i.e. effective segments = min(N, #sentences)).
Remove any double newlines before tokenizing.
"""
def split_into_segments(text, n):
    # Clean double newlines
    text = text.replace('\n\n', ' ')
    sentences = sent_tokenize(text)
    total = len(sentences)
    # If fewer sentences than requested, one segment per sentence
    effective_n = min(n, total) if total > 0 else 0

    segments = []
    if effective_n > 0:
        q, r = divmod(total, effective_n)
        idx = 0
        for i in range(effective_n):
            size = q + (1 if i < r else 0)
            segment = " ".join(sentences[idx:idx + size])
            segments.append(segment)
            idx += size
    return segments

"""
Process one file: read, clean, split into segments, and tag each with a unique ID.
"""
def process_file_with_range(args):
    filepath, n_segments, start_id = args
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    segments = split_into_segments(text, n_segments)
    filename = os.path.basename(filepath)
    entries = [
        {"id": start_id + i, "filename": filename, "text": seg}
        for i, seg in enumerate(segments)
    ]
    return entries

"""
Main execution: discover files, prepare tasks with correct start IDs,
run in parallel, and write JSONL + log progress.
"""
def main(input_path, output_dir, max_segments):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "segments.jsonl")
    log_file = os.path.join(output_dir, "process.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"Start: input={input_path}, output_dir={output_dir}, max_segments={max_segments}")

    # Collect .txt files
    if os.path.isfile(input_path):
        files = [input_path]
    elif os.path.isdir(input_path):
        files = sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith('.txt')
        )
    else:
        logging.error(f"Invalid input path: {input_path}")
        print(f"Error: {input_path} is not a valid file or directory.")
        return

    # Prepare tasks: (filepath, effective_segments, start_id)
    tasks = []
    current_id = 1
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read().replace('\n\n', ' ')
        total_sent = len(sent_tokenize(text))
        if total_sent == 0:
            logging.warning(f"Skipping empty file: {os.path.basename(fpath)}")
            continue
        eff_n = min(max_segments, total_sent)
        tasks.append((fpath, eff_n, current_id))
        logging.info(f"Queued {os.path.basename(fpath)}: {eff_n} segments (IDs {current_id}-{current_id+eff_n-1})")
        current_id += eff_n

    all_entries = []
    # Parallel processing with progress bar
    with Pool(processes=min(cpu_count(), len(tasks))) as pool:
        for entries in tqdm(
            pool.imap_unordered(process_file_with_range, tasks),
            total=len(tasks),
            desc="Processing files"
        ):
            if entries:
                fname = entries[0]['filename']
                s, e = entries[0]['id'], entries[-1]['id']
                logging.info(f"Processed {fname}: {len(entries)} segments (IDs {s}-{e})")
                all_entries.extend(entries)

    # Write all segments to JSONL
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for entry in all_entries:
            json.dump(entry, out_f)
            out_f.write('\n')

    logging.info(f"Finished: wrote {len(all_entries)} segments to {output_file}")
    print(f"✅ Done! Written {len(all_entries)} segments to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split .txt file(s) into up to N sentence-based segments, exact count if fewer sentences, output JSONL with logging and progress bars."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="76_VHA_YouTube_Corpus",
        help="Input .txt file or directory (default: %(default)s)"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="output",
        help="Directory for output JSONL and log (default: %(default)s)"
    )
    parser.add_argument(
        "-n", "--n",
        type=int,
        dest="max_segments",
        default=100,
        help="Maximum number of segments per file (default: %(default)s); actual segments = min(n, #sentences)"
    )
    args = parser.parse_args()
    main(args.input, args.output_dir, args.max_segments)
