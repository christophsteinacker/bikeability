import itertools as it
from bikeability_optimisation.helper.plot_helper import calc_scale
from bikeability_optimisation.main.plot import compare_cities

cities = ['Hamburg', 'Oslo']
saves = {'Hamburg': 'hh', 'Oslo': 'oslo'}

base_city = 'Hamburg'

comp_folder = 'data/plot_data/'
plot_folder = 'plots/results/'

minmode = [1]
rev = [False]
modes = list(it.product(rev, minmode))

colors = ['royalblue', 'orangered']
comp_color = {city: colors[idx] for idx, city in enumerate(cities)}

print('Calculating scaling factor')
scale_x = calc_scale(base_city, cities, saves, comp_folder, modes[0])
for m in modes:
    print('Comparing cities.')
    compare_cities(cities=cities, saves=saves, mode=m, color=comp_color,
                   data_folder=comp_folder, plot_folder=plot_folder,
                   scale_x=scale_x, base_city=base_city, titles=True,
                   legends=True, plot_format='png')
