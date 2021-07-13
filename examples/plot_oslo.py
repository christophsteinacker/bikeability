from params import params
from paths import paths
from bikeability_optimisation.main.plot import plot_city

save = 'oslo'
city = 'Oslo'

plot_city(city, save, paths=paths, params=params)
