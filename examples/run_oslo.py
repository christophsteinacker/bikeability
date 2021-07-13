from params import params
from paths import paths
from bikeability_optimisation.main.data import prep_city
from bikeability_optimisation.main.algorithm import run_simulation
from bikeability_optimisation.main.plot import plot_city
from multiprocessing import Pool, cpu_count
from functools import partial

city = 'Oslo'
save = 'oslo'

nominatim_name = 'Oslo, Norway'
nominatim_result = 2

input_csv = f'data/cleaned_data/{save}_cleaned.csv'

prep_city(city_name=city, save_name=save, input_csv=input_csv,
          nominatim_name=nominatim_name, nominatim_result=nominatim_result,
          trunk=False, consolidate=True, tol=35,
          by_bbox=False, by_city=False, by_polygon=True,
          paths=paths, params=params)

print('Prepared data for calculations. Starting Calculations now.')

modes = params["modes"]

fnc = partial(run_simulation, city, save, params, paths)

p = Pool(processes=min(cpu_count(), len(modes)))
data = p.map(fnc, modes)
p.close()

print('Calculations complete! Now Plotting.')

plot_city(city, save, paths=paths, params=params)
