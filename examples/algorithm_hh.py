from bikeability_optimisation.main.algorithm import run_simulation
from multiprocessing import Pool, cpu_count
from functools import partial
import itertools as it

city = 'Hamburg'
save = 'hh'

input_folder = 'data/input_data/{}/'.format(save)
output_folder = 'data/output_data/{}/'.format(save)
log_folder = 'logs/{}/'.format(save)

minmode = [0, 1, 2]
rev = [False, True]

modes = list(it.product(rev, minmode))

fnc = partial(run_simulation, city, save, input_folder, output_folder,
              log_folder)

p = Pool(processes=min(cpu_count(), len(modes)))
data = p.map(fnc, modes)
p.close()

print('Run complete!')
