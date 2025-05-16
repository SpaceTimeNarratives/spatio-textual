#!/usr/bin/env python3
"""
spatial_entity_extraction_cli.py

A CLI tool to annotate text segments with spatial entities and verb relations.
"""
import argparse
import json
import os
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import spacy
from geonamescache import GeonamesCache


def load_spacy_model(model_name: str = 'en_core_web_trf') -> spacy.Language:
    """
    Load and configure a spaCy model, adding entity merging and an entity ruler.
    """
    nlp = spacy.load(model_name)
    nlp.add_pipe('merge_entities')
    nlp.add_pipe('entity_ruler', before='ner')
    return nlp


class PlaceNames:
    """
    Base class for loading geographic names and classifying place-related entities.
    Loads cities, states, countries, continents, and custom resource lists.
    """
    def __init__(self, resources_dir: str = None):
        self.gc = GeonamesCache()
        self._load_geo_data()
        self._load_lists(resources_dir)

    def _load_geo_data(self):
        """
        Retrieve and sort names for cities, US states, countries, continents.
        """
        cities = self.gc.get_cities()
        states = self.gc.get_us_states()
        countries = self.gc.get_countries()
        continents = self.gc.get_continents()

        self.city_names = sorted({c['name'] for c in cities.values()} | {'New York'}, key=len, reverse=True)
        self.state_names = sorted({s['name'] for s in states.values()}, key=len, reverse=True)
        self.country_names = sorted({c['name'] for c in countries.values()} | 
                                     {'America', 'the United States', 'Czechoslovakia'},
                                     key=len, reverse=True)
        self.continent_names = sorted({c['name'] for c in continents.values()}, key=len, reverse=True)

    def _load_lists(self, resources_dir: str = None):
        """
        Load custom resource lists for additional entity labels:
          - combined_geonouns.txt
          - cleaned_holocaust_camps.txt
          - ambiguous_cities.txt
          - non_verbal.txt
          - family_terms.txt
        Missing files result in empty lists.
        """
        base = Path(resources_dir) if resources_dir else Path()

        def read_list(fname: str):
            path = base / fname
            if not path.exists():
                return []
            return sorted({l.strip() for l in path.read_text().splitlines() if l.strip()},
                          key=len, reverse=True)

        self.geonouns = read_list('combined_geonouns.txt')
        self.camps = read_list('cleaned_holocaust_camps.txt')
        self.ambiguous_cities = read_list('ambiguous_cities.txt')
        self.non_verbal = read_list('non_verbal.txt')
        self.family = read_list('family_terms.txt')

    def classify(self, text: str, label: str) -> str:
        """
        Classify place-like entities into CONTINENT, COUNTRY, US-STATE,
        CITY, CAMP, or default PLACE.
        """
        if label in {'FAC', 'GPE', 'LOC', 'ORG'}:
            if text in self.continent_names:
                return 'CONTINENT'
            if text in self.country_names:
                return 'COUNTRY'
            if text in self.state_names:
                return 'US-STATE'
            if text in self.city_names:
                return 'CITY'
            if text in self.camps:
                return 'CAMP'
            return 'PLACE'
        return label


class Annotator(PlaceNames):
    """
    Performs entity annotation and verb extraction using spaCy.
    Inherits classification methods from PlaceNames.
    """
    def __init__(self, nlp: spacy.Language, resources_dir: str = None):
        super().__init__(resources_dir)
        self.nlp = nlp

    def extract_verbs(self, doc: spacy.tokens.Doc) -> list:
        """
        From each sentence, extract verbs with their subject and object.
        """
        data = []
        for sent in doc.sents:
            for token in sent:
                if token.pos_ == 'VERB':
                    subj = [c.text for c in token.children if c.dep_ in ('nsubj', 'nsubjpass')]
                    obj = [c.text for c in token.children if c.dep_ == 'dobj'] + \
                          [gc.text for prep in token.children if prep.dep_ == 'prep'
                           for gc in prep.children if gc.dep_ == 'pobj']
                    data.append({
                        'verb': token.text,
                        'subject': subj[0] if subj else '',
                        'object': obj[0] if obj else '',
                        'sentence': sent.text
                    })
        return data

    def annotate(self, text: str) -> dict:
        """
        Annotate a text segment:
          - Extract and classify entities
          - Extract verb relations
        Returns {'entities': {char_offset: {text, tag}}, 'verb_data': [...]}
        """
        doc = self.nlp(text)
        entities = {}
        for ent in doc.ents:
            if ent.label_ in {
                'PERSON','FAC','GPE','LOC','ORG','DATE','TIME','EVENT',
                'QUANTITY','GEONOUN','NON-VERBAL','FAMILY'
            }:
                tag = ent.label_
                if ent.label_ in {'FAC','GPE','LOC','ORG'}:
                    tag = self.classify(ent.text, ent.label_)
                entities[ent.start_char] = {'text': ent.text, 'tag': tag}

        verb_data = self.extract_verbs(doc)
        return {'entities': entities, 'verb_data': verb_data}


def process_record(record: dict, annotator: Annotator) -> dict:
    """
    Annotate a single record dictionary with 'text'.
    """
    text = record.get('text', '')
    result = annotator.annotate(text)
    record.update(result)
    return record


def process_file(in_path: Path, out_path: Path, annotator: Annotator, workers: int):
    """
    Read input file (JSONL), annotate each record, and write JSONL output.
    Supports multiprocessing.
    """
    with open(in_path, 'r', encoding='utf-8') as f_in, \
         open(out_path, 'w', encoding='utf-8') as f_out:
        records = [json.loads(line) for line in f_in]
        if workers > 1:
            with Pool(workers) as pool:
                func = partial(process_record, annotator=annotator)
                for rec in tqdm(pool.imap(func, records), total=len(records), desc=in_path.name):
                    f_out.write(json.dumps(rec) + '\n')
        else:
            for rec in tqdm(records, desc=in_path.name):
                annotated = process_record(rec, annotator)
                f_out.write(json.dumps(annotated) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Spatial Entity Extraction CLI')
    parser.add_argument('-i', '--input', required=True, nargs='+',
                        help='Input JSONL files to annotate')
    parser.add_argument('-o', '--output', required=True,
                        help='Directory for annotated outputs')
    parser.add_argument('-m', '--model', default='en_core_web_trf',
                        help='spaCy model name')
    parser.add_argument('-r', '--resources', default=None,
                        help='Path to resource files')
    parser.add_argument('-w', '--workers', type=int, default=1,
                        help='Number of parallel workers')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    nlp = load_spacy_model(args.model)
    annotator = Annotator(nlp, args.resources)

    for infile in args.input:
        in_path = Path(infile)
        out_name = in_path.stem + '_annotated.jsonl'
        out_path = Path(args.output) / out_name
        process_file(in_path, out_path, annotator, args.workers)


if __name__ == '__main__':
    main()
