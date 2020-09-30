from bikeability_optimisation.main.data import prep_city
from bikeability_optimisation.main.algorithm import run_simulation
from bikeability_optimisation.main.plot import plot_city
from multiprocessing import Pool, cpu_count
from functools import partial
import itertools as it

city = 'Oslo'
save = 'oslo'

nominatim_name = 'Oslo, Norway'
nominatim_result = 2

input_csv = 'data/cleaned_data/{}_cleaned.csv'.format(save)
poly_folder = 'data/cropped_areas/'
polygon = '{}{}.json'.format(poly_folder, save)
preparation_plots = 'plots/preparation/'

input_algorithm = 'data/input_data/{}/'.format(save)
output_algorithm = 'data/output_data/{}/'.format(save)
log_folder = 'logs/{}/'.format(save)

minmode = [1]
rev = [False, True]
modes = list(it.product(rev, minmode))

result_plots = 'plots/results/{}/'.format(save)
result_comp = 'data/plot_data/'

# [width, height]
plot_bbox_size = [20, 20]
plot_city_size = [20, 20]
plot_size = [20, 20]

prep_city(city, save, input_csv, input_algorithm, polygon, preparation_plots,
          nominatim_name=nominatim_name, nominatim_result=nominatim_result,
          trunk=False, consolidate=True, tol=35,
          by_bbox=False, plot_bbox_size=plot_bbox_size, by_city=False,
          plot_city_size=plot_city_size, by_polygon=True, plot_size=plot_size,
          cached_graph=False, cached_graph_folder=input_algorithm,
          cached_graph_name=save)

fnc = partial(run_simulation, city, save, input_algorithm, output_algorithm,
              log_folder)

p = Pool(processes=min(cpu_count(), len(modes)))
data = p.map(fnc, modes)
p.close()

plot_city(city=city, save=save, polygon_folder=poly_folder,
          input_folder=input_algorithm, output_folder=output_algorithm,
          comp_folder=result_comp, plot_folder=result_plots, modes=modes,
          comp_modes=False, plot_evo=False, correct_area=True,
          plot_format='png')
