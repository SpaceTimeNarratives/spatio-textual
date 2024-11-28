
# Utility functions

import re
import spacy

# Sort a list by length, ensuring uniqueness
def sortbylen(lst):
    return sorted(set(lst), key=len, reverse=True)

def distribute_into_segments(strings, num_segments=100):
  # Calculate the size of each segment
  segment_size = len(strings) // num_segments
  remainder = len(strings) % num_segments

  segments, start = [], 0

  for i in range(num_segments):
      # Calculate the end index for this segment
      end = start + segment_size + (1 if i < remainder else 0)
      # Append the segment to the list
      segments.append(strings[start:end])
      # Update the start index for the next segment
      start = end
  return segments

def combine_speakers_with_next(strings):
  combined_list = []
  skip_next = False

  # Regular expression to identify speaker abbreviations (e.g., 'INT:', 'HP:')
  speaker_pattern = re.compile(r'^[A-Z]{2,4}:?\s*$')

  for i in range(len(strings) - 1):
    if skip_next:
        skip_next = False
        continue

    current_str = strings[i]
    next_str = strings[i + 1]

    if speaker_pattern.match(current_str):
        # Combine the current speaker string with the next one
        combined_list.append(current_str + ' ' + next_str)
        skip_next = True  # Skip the next string since it's already combined
    else:
        combined_list.append(current_str)

  # If the last string wasn't combined with anything, add it as is
  if not skip_next:
    combined_list.append(strings[-1])

  return combined_list


#Load spaCy model and add the geonoun pattern to the entity ruler
nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('merge_entities')

# Add the 'entity_ruler' to the pipeline before the NER module
ruler = nlp.add_pipe("entity_ruler", before='ner')

# Function to read patterns from a file and create a list of patterns for the EntityRuler
def load_patterns(label, filename):
    with open(f"resources/{filename}") as f:
        return [{"label": label, "pattern": line.strip()} for line in f if line.strip()]

# Load and add patterns from files
files_and_labels = [
    ('combined_geonouns.txt', 'GEONOUN'),
    ('non_verbals.txt', 'NON-VERBAL'),
    ('family_relationships.txt', 'FAMILY')
]
# Add all patterns to the ruler
patterns = []
for filename, label in files_and_labels:
    patterns.extend(load_patterns(label, filename))
ruler.add_patterns(patterns)
