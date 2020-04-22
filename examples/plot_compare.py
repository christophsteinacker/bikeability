import itertools as it
from bikeability_optimisation.main.plotting import compare_cities

cities = ['Hamburg', 'Oslo']
saves = {'Hamburg': 'hh', 'Oslo': 'oslo'}

comp_folder = 'data/plot_data/'
plot_folder = 'plots/results/'

minmode = [1]
rev = [False]
mode = list(it.product(rev, minmode))

colors = ['royalblue', 'orangered']
comp_color = {city: colors[idx] for idx, city in enumerate(cities)}

for m in mode:
    compare_cities(cities=cities, saves=saves, mode=m, color=comp_color,
                   data_folder=comp_folder, plot_folder=plot_folder,
                   plot_format='png')
