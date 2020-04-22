import itertools as it
from bikeability_optimisation.main.plot import plot_city

city = 'Hamburg'
save = 'hh'

input_folder = 'data/input_data/{}/'.format(save)
output_folder = 'data/output_data/{}/'.format(save)

plot_folder = 'plots/results/{}/'.format(save)
comp_folder = 'data/plot_data/'

minmode = [1]
rev = [False, True]
modes = list(it.product(rev, minmode))

plot_city(city, save, input_folder, output_folder, comp_folder,
          plot_folder, modes, comp_modes=True, plot_format='png')
