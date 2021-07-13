from params import params
from paths import paths
from bikeability_optimisation.main.plot import plot_city

save = 'hh'
city = 'Hamburg'

plot_city(city, save, paths=paths, params=params)
