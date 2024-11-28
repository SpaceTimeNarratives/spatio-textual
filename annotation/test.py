from geonamescache import GeonamesCache as gc

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
    isCityAmbiguous = lambda self, city: (True, [_city for _, _city in self.cities.items() 
                                                 if _city['name'].lower() == city.lower()]
                                          ) if city in self.ambiguous_cities else (False, 
                                                                f"'{city}' is not ambiguous")
# ===========End PlaceNames Class============

if __name__ == "__main__":
  placenames = PlaceNames()  
  print('Done!')
  