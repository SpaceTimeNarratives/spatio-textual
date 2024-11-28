#@title Setup for the NLP and LLM pipeline

from geonamescache import GeonamesCache as gc
import os, re, json, time
import spacy
from tqdm import tqdm
from collections import OrderedDict
from .utils import sortbylen


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


# @title #### Class `PlaceNames`:<br>For named places and types (CONTINENT, COUNTRY, CITY, CAMPS and PLACE if not one of the other types)
class PlaceNames:
    def __init__(self):
      self.additional_cities = ['New York'] #cities not in GeoNames or Aliases
      self.additional_countries = ['America', 'the United States','Czechoslovakia'] #countries not in GeoNames or Aliases
      self.cities, self.city_names = self.__get_cities()
      self.us_states, self.us_state_names = self.__get_us_states()
      self.countries, self.country_names = self.__get_countries()
      self.continents, self.continent_names = self.__get_continents()
      self.camps = self.__get_camps()
      self.geonouns = self.__get_geonouns()
      self.ambiguous_cities = self.__get_ambiguous_cities()

    # read source files
    def __read_source_file(self, source_file):
      return open(source_file, 'r', encoding='utf-8', errors='replace').read().strip().split('\n')

    # city details and names
    def __get_cities(self):
      __cities = {i:{'geonameid':detail['geonameid'], 'name':detail['name'].replace("'",'’'),
             'latitude':float(detail['latitude']), 'longitude':float(detail['longitude']),
             'countrycode':detail['countrycode']} for i, (_, detail) in enumerate(gc().get_cities().items())}
      __names = [city['name'] for _, city in __cities.items()]
      __names.extend(self.additional_cities)
      return __cities, sortbylen(__names)

    # US states details and names
    def __get_us_states(self):
      __us_states = {i:{'geonameid':detail['geonameid'],'name':detail['name'].replace("'",'’'),'code':detail['code']}
              for i, (_, detail) in enumerate(gc().get_us_states().items())}
      __names = sortbylen([us_state['name'] for _, us_state in __us_states.items()])
      return __us_states, __names

    # country details and names
    def __get_countries(self):
      __countries = {i:{'geonameid':detail['geonameid'], 'iso': detail['iso'], 'name':detail['name'].replace("'",'’'),
        'capital':detail['capital'].replace("'",'’'), 'continentcode':detail['continentcode'], 'neighbours':detail['neighbours']}
        for i, (_, detail) in enumerate(gc().get_countries().items())}
      __names = [country['name'] for _, country in __countries.items()]
      __names.extend(self.additional_countries)
      return __countries, sortbylen(__names)

    # continent details and names
    def __get_continents(self):
      __continents = {i:{'geonameid':detail['geonameId'], 'name':detail['name'].replace("'",'’'),
                         'continentcode':detail['continentCode'], 'bbox_north':detail['bbox']['north'],
                         'bbox_south':detail['bbox']['south'], 'bbox_east':detail['bbox']['east'],
                         'bbox_west':detail['bbox']['west']} for i, (_, detail) in enumerate(gc().get_continents().items())}
      __names = sortbylen([continent['name'] for _, continent in __continents.items()])
      return __continents, __names

  # Concentration camps
    def __get_camps(self, srcfile=None):
      source_file = srcfile if srcfile else 'cleaned_holocaust_camps.txt'
      __camps = self.__read_source_file(source_file)
      if __camps: return sortbylen([name for name in __camps if name not in [country['name']
                                                for _, country in self.countries.items()]])
      else:
        print(f"Error: Reading file '{source_file}'.")
        return None

  # Geographical feature names
    def __get_geonouns(self, srcfile=None):
      source_file = srcfile if srcfile else 'combined_geonouns.txt'
      __geonouns = self.__read_source_file(source_file)
      if __geonouns: return sortbylen(__geonouns)
      else:
        print(f"Error: Reading file '{source_file}'.")
        return None

  # Get ambiguous cities
    def __get_ambiguous_cities(self, srcfile=None):
      source_file = srcfile if srcfile else 'ambiguous_cities.txt'
      __ambiguous_cities = self.__read_source_file(source_file)
      if __ambiguous_cities: return sortbylen(__ambiguous_cities)
      else:
        print(f"Error: Reading file '{source_file}'.")
        return None

  # Check Ambiguous Cities
    isCityAmbiguous = lambda self, city: (True, [_city for _, _city in self.cities.items() if _city['name'].lower() == city.lower()]
                                          ) if city in self.ambiguous_cities else (False,f"{city} is not ambiguous: {[_city for _, _city in self.cities.items() if _city['name'].lower() == city.lower()][0]}")
# ===========End PlaceNames Class============


# @title #### Class `Annotator`:<br>For actual annotations entities and emotion classification
class Annotator(PlaceNames):
    def __init__(self, **kwargs): #kwargs = ['text', 'model']
      self.resources_url= "https://raw.githubusercontent.com/SpaceTimeNarratives/demo/main/resources/"
      self.__download_resources()
      super().__init__()
      self.text             = kwargs['text'] if 'text' in kwargs else None
      self.texts            = kwargs['texts'] if 'texts' in kwargs else None
      self.nlp              = kwargs['nlp_model'] if 'nlp_model' in kwargs else None
      self.output_dir       = 'output'
      self.emotion_model    = kwargs['emotion_model'] if 'emotion_model' in kwargs else None
      self.sentiment_model  = kwargs['sentiment_model'] if 'sentiment_model' in kwargs else None
      self.__BG_COLOR       = {'CITY':'#feca74','COUNTRY':'#f0b6de','CONTINENT':'#e4e7d2',
                               'US-STATE':'#feca74','CAMP':'#b3d6f2','GEONOUN': '#9cc9cc',
                               'DATE':'#c7f5a9', 'TIME':'#a9f5bc','PLACE':'#e4e7d2',
                               'EVENT':'#e0aedd', 'NON-VERBAL':'#ba8fc7', 'FAMILY':'#c9b99d'}
      self.entity_tags      = self.__BG_COLOR.keys()

      self.sentiment_scores  = None
      self.emotion_scores    = None

    # Download resource files()
    def __download_resources(self):
      for res in ['cleaned_holocaust_camps.txt','combined_geonouns.txt',
                  'ambiguous_cities.txt', 'ht_non_verbals.txt',
                  'family_relationships.txt', 'ht_998_file_list.txt']:
        if not os.path.exists(res):
          os.system(f"wget -q {self.resources_url}{res}")
          print(f"{res} successfully downloaded.")

    # Below are the helper functions for merging entities
    def __merge_entities(self, first_ents, second_ents):
      return dict(OrderedDict(sorted({**second_ents, **first_ents}.items())))

    # Merge adjacent similar entities
    def __join_near_similar_ents(self, ent_dict, tag):
      return {i:(ent[0]+' '+ent_dict[i+len(ent[0])+1][0], tag)
              for i, ent in ent_dict.items() if ent[1]==tag and i+len(ent[0])+1 in ent_dict}

    def __convert_place_entities(self, place):
      name, tag = place
      if tag in ['FAC','GPE','LOC','ORG']:
        if name in self.continent_names: return name, 'CONTINENT'
        elif name in self.country_names: return name, 'COUNTRY'
        elif name in self.us_state_names: return name, 'US-STATE'
        elif name in self.city_names: return name, 'CITY'
        elif name in self.camps: return name, 'CAMP'
        else: return name, 'PLACE'
      return name, tag

    # annotate entities in text
    def annotate_text(self, text=None):
      if text: self.text = text
      if self.text: doc = self.nlp(self.text)
      else: return f"Error: 'Annotator' has no text to process!"

      __ent_details = {token.idx:(self.text[token.idx:token.idx+len(token)],
         token.ent_type_, token.pos_) for token in doc if token.ent_type_ in
          ['PERSON','FAC','GPE','LOC', 'ORG','DATE','TIME','EVENT', 'QUANTITY', 'GEONOUN', 'NON-VERBAL','FAMILY']}

      # enforce only 'GEONOUNS' pos-tagged as 'NOUN'
      __ent_details= {i:detail for i, detail in __ent_details.items() if detail[:2]!='GEONOUN' or (detail[:2]=='GEONOUN' and detail[:3]=='NOUN')}

      #join near similar ents e.g. "concentration:GEONOUN", "camp:GEONOUN" --> "concentration camp:GEONOUN"
      __ent_details= self.__merge_entities(self.__join_near_similar_ents(__ent_details, 'GEONOUN'), __ent_details)
      return {i:self.__convert_place_entities(detail[:2]) for i, detail in __ent_details.items()}

    # extract entities from text
    def annotate_texts(self, texts=None):
      if texts: self.texts = texts
      docs = []
      if self.texts:
        for docid, doc in enumerate(pbar := tqdm(self.nlp.pipe(self.texts), desc=f"Annotating segment 0")):
          pbar.set_description(f"Annotating segment {docid}")
          docs.append(doc)
      else:
         return f"Warning: 'Annotator' has no text to process!"

      ent_classes = ['PERSON','FAC','GPE','LOC', 'ORG','DATE','TIME','EVENT',
                     'QUANTITY', 'GEONOUN', 'NON-VERBAL','FAMILY']
      annotations={}
      print("Formatting entities and labelling emotions...")
      pbar = tqdm(enumerate(docs))
      for docid, doc in pbar:
        pbar.set_description(f"-Labelling segment {docid:02d}")
        __ent_detail = {token.idx:(doc.text[token.idx:token.idx+len(token)],
         token.ent_type_, token.pos_) for token in doc if token.ent_type_ in ent_classes}

        # enforce only 'GEONOUNS' pos-tagged as 'NOUN'
        __ent_detail = {i:detail for i, detail in __ent_detail.items()
                        if detail[:2]!='GEONOUN' or (detail[:2]=='GEONOUN' and detail[:3]=='NOUN')}

        #join near similar ents e.g. "concentration:GEONOUN", "camp:GEONOUN" --> "concentration camp:GEONOUN"
        __ent_detail= self.__merge_entities(self.__join_near_similar_ents(__ent_detail, 'GEONOUN'), __ent_detail)

        annotations[docid] = {
            'text': doc.text,
            'entities': {i:self.__convert_place_entities(detail[:2]) for i, detail in __ent_detail.items()},
            }
      print("Done!")
      return annotations
# ===========End Annotator Class===========

if __name__ == "__main__":
  annotator = Annotator(nlp_model=nlp)
  
  print('Done!')
  