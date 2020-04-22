from bikeability_optimisation.main.algorithm import run_simulation
from multiprocessing import Pool
from functools import partial
import itertools as it

city = 'Hamburg'
save = 'hh'

input_folder = 'data/input_data/{}/'.format(save)
output_folder = 'data/output_data/{}/'.format(save)
log_folder = 'logs/{}/'.format(save)

minmode = [1]
rev = [False, True]

mode = list(it.product(rev, minmode))

fnc = partial(run_simulation, city, save, input_folder, output_folder,
              log_folder)

p = Pool(processes=2)
data = p.map(fnc, mode)
