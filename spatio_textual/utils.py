import spacy
from pathlib import Path
from geonamescache import GeonamesCache

def load_spacy_model(model_name: str = 'en_core_web_trf') -> spacy.Language:
    nlp = spacy.load(model_name)
    nlp.add_pipe('merge_entities')
    nlp.add_pipe('entity_ruler', before='ner')
    return nlp

class PlaceNames:
    def __init__(self, resources_dir: str = None):
        self.gc = GeonamesCache()
        self._load_geo_data()
        self._load_lists(resources_dir)

    def _load_geo_data(self):
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
        base = Path(resources_dir) if resources_dir else Path(__file__).parent / 'resources'

        def read_list(fname: str):
            path = base / fname
            if not path.exists():
                return []
            return sorted({l.strip() for l in path.read_text().splitlines() if l.strip()}, key=len, reverse=True)

        self.geonouns = read_list('combined_geonouns.txt')
        self.camps = read_list('cleaned_holocaust_camps.txt')
        self.ambiguous_cities = read_list('ambiguous_cities.txt')
        self.non_verbal = read_list('non_verbal.txt')
        self.family = read_list('family_terms.txt')

    def classify(self, text: str, label: str) -> str:
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
    def __init__(self, nlp: spacy.Language, resources_dir: str = None):
        super().__init__(resources_dir)
        self.nlp = nlp

    def extract_verbs(self, doc: spacy.tokens.Doc) -> list:
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
