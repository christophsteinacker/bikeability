from bikeability_optimisation.main.data import prep_city
from bikeability_optimisation.main.algorithm import run_simulation
from bikeability_optimisation.main.plot import plot_city
from multiprocessing import Pool
from functools import partial
import itertools as it

city = 'Oslo'
save = 'oslo'

nominatim_name = 'Oslo, Norway'
nominatim_result = 2

input_csv = 'data/cleaned_data/{}_cleaned.csv'.format(save)
polygon = 'data/cropped_areas/{}.json'.format(save)
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

prep_city(city, save, nominatim_name, nominatim_result, input_csv,
          input_algorithm, polygon, preparation_plots, trunk=False,
          by_bbox=True, plot_bbox_size=plot_bbox_size,
          by_city=True, plot_city_size=plot_city_size,
          by_polygon=True, plot_size=plot_size)

fnc = partial(run_simulation, city, save, input_algorithm, output_algorithm,
              log_folder)
p = Pool(processes=2)
data = p.map(fnc, modes)

plot_city(city, save, input_algorithm, output_algorithm, result_comp,
          result_plots, modes, comp_modes=True, plot_format='png')
