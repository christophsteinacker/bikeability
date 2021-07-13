from params import params
from paths import paths
from bikeability_optimisation.main.data import prep_city

nominatim_name = 'Oslo, Norway'
nominatim_result = 2

city = 'Oslo'
save = 'oslo'

input_csv = f'data/cleaned_data/{save}_cleaned.csv'

prep_city(city_name=city, save_name=save, input_csv=input_csv,
          nominatim_name=nominatim_name, nominatim_result=nominatim_result,
          trunk=False, consolidate=True, tol=35,
          by_bbox=False, by_city=False, by_polygon=True,
          paths=paths, params=params)
