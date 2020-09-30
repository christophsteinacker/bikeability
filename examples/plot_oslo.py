import itertools as it
from bikeability_optimisation.main.plot import plot_city

save = 'oslo'
city = 'Oslo'

input_folder = 'data/input_data/{}/'.format(save)
output_folder = 'data/output_data/{}/'.format(save)

plot_folder = 'plots/results/{}/'.format(save)
comp_folder = 'data/plot_data/'

poly_folder = 'data/cropped_areas/'

minmode = [1]
rev = [False]
modes = list(it.product(rev, minmode))

plot_city(city=city, save=save, polygon_folder=poly_folder,
          input_folder=input_folder, output_folder=output_folder,
          comp_folder=comp_folder, plot_folder=plot_folder, modes=modes,
          comp_modes=False, plot_evo=False, correct_area=True, titles=True,
          legends=True, plot_format='png')
