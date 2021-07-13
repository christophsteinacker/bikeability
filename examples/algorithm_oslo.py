from params import params
from paths import paths
from bikeability_optimisation.main.algorithm import run_simulation
from multiprocessing import Pool, cpu_count
from functools import partial

city = 'Oslo'
save = 'oslo'

modes = params["modes"]

fnc = partial(run_simulation, city, save, params, paths)

p = Pool(processes=min(cpu_count(), len(modes)))
data = p.map(fnc, modes)
p.close()

print('Run complete!')
