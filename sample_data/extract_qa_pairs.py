import re
import json
import argparse

def parse_transcript_to_qa_pairs(text):
    """
    Parse a transcript string into question-answer pairs.
    
    Lines spoken by INT or CREW are treated as questions.
    Lines spoken by others (e.g., EB, relatives) are treated as answers.
    Pairs are generated sequentially: a question is paired with the next valid answer.
    """
    # Regular expression to match speaker lines, e.g., "INT: ..." or "EB: ..."
    speaker_pattern = re.compile(r'^([A-Z]{2,5}):\s*(.*)', re.MULTILINE)
    
    # Extract all speaker-line matches
    matches = list(speaker_pattern.finditer(text))

    qa_pairs = []
    question_speakers = {"INT", "CREW"}  # Speakers whose lines will be treated as questions
    i = 0

    while i < len(matches):
        speaker_i = matches[i].group(1)
        text_i = matches[i].group(2).strip()
        full_i = f"{speaker_i}: {text_i}"

        # If this is a question speaker
        if speaker_i in question_speakers:
            question = full_i
            answer = ""

            # Look for the next non-question speaker to get the answer
            j = i + 1
            while j < len(matches):
                speaker_j = matches[j].group(1)
                text_j = matches[j].group(2).strip()
                if speaker_j not in question_speakers:
                    answer = f"{speaker_j}: {text_j}"
                    break
                j += 1

            qa_pairs.append({"question": question, "answer": answer})
            i = j if j > i else i + 1  # Skip to next unprocessed match
        else:
            i += 1  # Skip answer-only lines not preceded by a question

    return qa_pairs

def save_qa_to_jsonl(pairs, output_file):
    """
    Save question-answer pairs to a JSONL file.
    Each line contains a JSON object with 'question' and 'answer' fields.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in pairs:
            json.dump(pair, f)
            f.write('\n')

def main():
    """
    Entry point for the CLI. Parses arguments, processes input, and writes output.
    """
    parser = argparse.ArgumentParser(
        description="Extract Q&A pairs from a Holocaust testimony transcript in text format."
    )
    parser.add_argument('--input', '-i', required=True, help="Path to input transcript .txt file.")
    parser.add_argument('--output', '-o', required=True, help="Path to output .jsonl file with Q&A pairs.")
    args = parser.parse_args()

    # Read input text
    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read()

    # Process text and save output
    qa_pairs = parse_transcript_to_qa_pairs(text)
    save_qa_to_jsonl(qa_pairs, args.output)
    
    print(f"✅ Extracted {len(qa_pairs)} Q&A pairs and saved to: {args.output}")

if __name__ == "__main__":
    main()
